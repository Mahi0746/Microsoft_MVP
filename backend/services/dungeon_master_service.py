from typing import Dict, List, Any, TypedDict, Annotated, Union
import operator
import json
from datetime import datetime
import structlog
import httpx
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from config_flexible import settings

logger = structlog.get_logger(__name__)

# --- State Definition ---
class DungeonMasterState(TypedDict):
    """State for the Therapy Game Adventure."""
    user_id: str
    session_id: str
    game_type: str  # e.g., 'shoulder_rehabilitation'
    difficulty_level: int  # 1-10
    fatigue: int  # 0-100
    narrative_history: List[str]  # History of the story so far
    current_exercise: Dict[str, Any]  # The exercise currently assigned
    last_performance_score: int  # Score from the last exercise (0-100)
    messages: List[Any]  # For LLM chat history
    express_mode: bool  # New field for Express Mode

# --- Service Class ---
# --- Service Class ---
class DungeonMasterService:
    _graph = None

    @classmethod
    def get_graph(cls):
        """Lazy load the graph."""
        if cls._graph is None:
            cls._graph = cls._build_graph()
        return cls._graph

    @classmethod
    def _build_graph(cls):
        """Builds the fully dynamic LangGraph StateGraph."""
        builder = StateGraph(DungeonMasterState)

        # Add Nodes representing the 'Council of AI'
        builder.add_node("storyteller", cls._storyteller_node)
        builder.add_node("motion_architect", cls._motion_architect_node)
        builder.add_node("safety_critic", cls._safety_critic_node)
        builder.add_node("performance_evaluator", cls._performance_evaluator_node)

        # Define Edges
        # Flow: Evaluator (if exists) -> Storyteller -> Motion Architect -> Safety Critic
        # If Safety Critic approves -> END
        # If Safety Critic rejects -> Loop back to Motion Architect
        
        # New Flow:
        # Entry -> Router -> (Storyteller -> Motion) OR (Motion)
        
        builder.add_node("mode_router", cls._mode_router_node)
        
        builder.set_entry_point("mode_router")
        
        builder.add_conditional_edges(
            "mode_router",
            cls._route_by_mode,
            {
                "story_mode": "storyteller",
                "express_mode": "motion_architect"
            }
        )
        
        builder.add_edge("storyteller", "motion_architect")
        builder.add_edge("motion_architect", "safety_critic")
        builder.add_conditional_edges(
            "safety_critic",
            cls._safety_router,
            {
                "approved": END,
                "rejected": "motion_architect"
            }
        )


        
        # If coming from evaluator, go to router to decide path
        builder.add_edge("performance_evaluator", "mode_router")

        return builder.compile()

    # --- Nodes ---

    @classmethod
    def _mode_router_node(cls, state: DungeonMasterState) -> Dict:
        """Passthrough node to decide routing"""
        return {}

    @classmethod
    def _route_by_mode(cls, state: DungeonMasterState) -> str:
        if state.get("express_mode", False):
            return "express_mode"
        return "story_mode"

    @classmethod
    def _performance_evaluator_node(cls, state: DungeonMasterState) -> Dict:
        """Analyzes the last exercise performance and adjusts difficulty/fatigue."""
        score = state.get("last_performance_score", 0)
        current_difficulty = state.get("difficulty_level", 3)
        current_fatigue = state.get("fatigue", 0)

        # Dynamic Fatigue Logic
        new_difficulty = current_difficulty
        narrative_update = ""

        if score > 85:
            new_difficulty = min(10, current_difficulty + 1)
            narrative_update = "User crushed the last challenge! The story should escalate in intensity."
        elif score < 60:
            new_difficulty = max(1, current_difficulty - 1)
            current_fatigue += 15
            narrative_update = "User is struggling or tired. The story should offer a respite or easier task."
        else:
            current_fatigue += 5
            narrative_update = "User is holding steady. Continue narrative flow."
        
        return {
            "difficulty_level": new_difficulty,
            "fatigue": current_fatigue,
            "messages": [SystemMessage(content=f"System Update: {narrative_update}")]
        }

    @classmethod
    async def _safe_llm_invoke(cls, messages: List[Any], temperature: float = 0.7) -> Any:
        """Helper to invoke LLM with OpenRouter fallback (HTTPX) on rate limit."""
        try:
            llm = ChatGroq(
                temperature=temperature,
                model_name=settings.groq_model or "llama-3.3-70b-versatile",
                groq_api_key=settings.groq_api_key
            )
            # Add stop for observation if needed (standard for agents, minimal harm here)
            return await llm.ainvoke(messages)
            
        except Exception as e:
             if "429" in str(e) or "Rate limit" in str(e):
                logger.warning(f"Groq Rate Limit in DungeonMaster. Switching to OpenRouter fallback (HTTP). Error: {e}")
                
                try:
                    headers = {
                        "Authorization": f"Bearer {settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://healthsync.ai",
                        "X-Title": "HealthSync AI"
                    }
                    
                    # Convert LangChain messages to OpenAI format
                    formatted_messages = []
                    for m in messages:
                        role = "user"
                        if hasattr(m, "type"):
                            if m.type == "system": role = "system"
                            elif m.type == "ai": role = "assistant"
                        formatted_messages.append({"role": role, "content": m.content})
                    
                    payload = {
                        "model": settings.openrouter_model,
                        "messages": formatted_messages,
                        "temperature": temperature
                    }

                    async with httpx.AsyncClient() as client:
                        or_resp = await client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=30.0
                        )
                        or_resp.raise_for_status()
                        content = or_resp.json()["choices"][0]["message"]["content"]
                        
                        return AIMessage(content=content)
                        
                except Exception as or_e:
                    logger.error(f"OpenRouter fallback failed: {or_e}")
                    raise e
             else:
                 raise e

    @classmethod
    async def _storyteller_node(cls, state: DungeonMasterState) -> Dict:
        """Generates the next unique segment of the adventure narrative."""
        
        game_type = state["game_type"]
        difficulty = state["difficulty_level"]
        history = state.get("narrative_history", [])[-2:]
        
        system_prompt = f"""You are the Master Storyteller for a unique 'Physio-RPG'.
        Context: The user is in a {game_type} session.
        Current Intensity: {difficulty}/10.
        Last Plot Points: {history}
        
        Task: Write the next 2 sentences of the adventure.
        - Be vivid and immersive.
        - Create a scenario that IMPLIES physical action, but do NOT name the exercise.
        - Example (Shoulder Rehab): "A massive vine hangs from the ancient tree, offering a way up the cliff." (Implies pulling/reaching).
        - Example (Knee Rehab): "The floor turns to lava! You must step carefully across the floating stones." (Implies lunges/high steps).
        """
        
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(state.get("messages", []))
        
        response = await cls._safe_llm_invoke(messages, temperature=0.8)
        story_segment = response.content

        return {
            "narrative_history": [story_segment],
            "messages": [AIMessage(content=story_segment)]
        }

    @classmethod
    async def _motion_architect_node(cls, state: DungeonMasterState) -> Dict:
        """Converts the story segment into a concrete physical exercise definition."""
        
        game_type = state["game_type"]
        difficulty = state["difficulty_level"]
        express_mode = state.get("express_mode", False)
        if express_mode:
            # FAST PATH: No story, direct exercise generation
            system_prompt = f"""You are a Physical Therapist.
            Create a quick {game_type} exercise.
            Difficulty: {difficulty}/10
            
            Return JSON ONLY:
            {{
                "name": "Standard Rehab Move",
                "description": "Clear instruction.",
                "duration": 30,
                "reps": 10
            }}
            """
            # Use cheaper model for express if desired, but versatile is fine
        else:
            # STORY PATH
            story = state["messages"][-1].content  # The story just generated
            system_prompt = f"""You are the Motion Architect. Your job is to translate a STORY into a THERAPY EXERCISE.
            
            Rehab Focus: {game_type}
            Story Scenario: "{story}"
            Difficulty: {difficulty}/10
            
            Task: Define a physical exercise that mimics the action in the story.
            - Uniqueness: Don't just say 'Arm Circles'. Give it a thematic name like 'Shield Wards'.
            - Motion: Must be safe and relevant to {game_type}.
            
            Return JSON ONLY:
            {{
                "name": "Thematic Name",
                "description": "How to do it (e.g. 'Raise arms slowly like climbing')",
                "duration": 30 (seconds),
                "reps": 10
            }}
            """
        
        response = await cls._safe_llm_invoke([SystemMessage(content=system_prompt)], temperature=0.5)
        # Simple extraction logic (in prod use JsonOutputParser)
        try:
            content = response.content
            # Strip markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            exercise_data = json.loads(content)
            return {"current_exercise": exercise_data}
        except:
            # Fallback if JSON fails
            return {"current_exercise": {"name": "Rest & Recover", "description": "Take a deep breath.", "duration": 15, "reps": 0}}

    @classmethod
    async def _safety_critic_node(cls, state: DungeonMasterState) -> Dict:
        """Validates if the generated exercise is safe for the user."""
        
        exercise = state["current_exercise"]
        game_type = state["game_type"]
        
        system_prompt = f"""You are the Safety Critic.
        Review this generated exercise for {game_type}:
        {json.dumps(exercise)}
        
        Is this safe and anatomically correct for {game_type}?
        - If YES, return "APPROVED".
        - If NO (e.g., it asks for squats during shoulder rehab, or dangerous ballistic moves), return "REJECTED".
        """
        
        response = await cls._safe_llm_invoke([SystemMessage(content=system_prompt)], temperature=0.1)
        decision = "approved" if "APPROVED" in response.content.upper() else "rejected"
        
        return {"messages": [SystemMessage(content=f"Safety Review: {decision}")]}

    @classmethod
    def _safety_router(cls, state: DungeonMasterState) -> str:
        """Route based on safety decision."""
        last_message = state["messages"][-1].content
        if "APPROVED" in last_message.upper():
            return "approved"
        return "rejected"

    @classmethod
    async def run_turn(cls, current_state: Dict) -> Dict:
        """Runs a single turn of the game loop."""
        graph = cls.get_graph()
        
        # Determine entry point
        if "last_performance_score" in current_state and current_state["last_performance_score"] is not None:
            # We need to manually inject the state to start at evaluator if we want, 
            # but since we set entry point to simple string, we can likely just run it.
            # LangGraph inputs are flexible.
            pass
            
        result_state = await graph.ainvoke(current_state, config={"recursion_limit": 10}) 
        return result_state
