# voice_agent_service.py - Manual ReAct Orchestrator

from langchain_groq import ChatGroq
from services.experts import (
    open_voice_consultant,
    get_current_medication,
    get_appointments,
    get_medical_reports,
    navigate_dashboard,
    answer_health_question,
    get_daily_health_plan
)
from config_flexible import settings
import httpx
import structlog
import json
import re

logger = structlog.get_logger(__name__)

class VoiceAgentService:
    _llm = None
    _tools = {}

    @classmethod
    def initialize(cls):
        """Initialize the Groq LLM and tools map."""
        if cls._llm:
            return

        try:
            # Initialize Groq LLM
            api_key = settings.groq_api_key
            if not api_key:
                logger.warning("No Groq API key found. Voice agent will not function strictly.")
                return

            cls._llm = ChatGroq(
                api_key=api_key,
                model_name=settings.groq_model or "llama-3.1-70b-versatile",
                temperature=0.0
            )

            # Register tools
            tools_list = [
                open_voice_consultant,
                get_current_medication,
                get_appointments,
                get_medical_reports,
                navigate_dashboard,
                answer_health_question,
                get_daily_health_plan
            ]
            cls._tools = {t.name: t for t in tools_list}
            
            logger.info("VoiceAgentService initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize VoiceAgentService: {e}")

    @classmethod
    async def process_voice_command(cls, user_input: str, user_context: dict = None, history: list = None) -> dict:
        """
        Process a text command from the voice interface using a manual ReAct loop.
        """
        if not cls._llm:
            cls.initialize()
            
        if not cls._llm:
            return {"type": "error", "message": "Voice service is not available (check API keys)."}

        if user_context is None:
            user_context = {"patient_id": "current_user", "role": "patient"}

        # Construct Tool Descriptions
        tool_desc = "\n".join([f"{t.name}: {t.description}" for t in cls._tools.values()])
        tool_names = ", ".join(cls._tools.keys())

        system_prompt = f"""
You are a smart healthcare voice assistant.
Your goal is to help patients navigate, view data, and get answers.

TOOLS:
------
{tool_desc}

FORMAT:
-------
To use a tool, please use the following format:
Thought: Do I need to use a tool? Yes
Action: [tool name]
Action Input: [tool input JSON string or simple text]
Observation: [tool output]

When you have a response for the user, or if checking the tool output is enough, you MUST use:
Thought: Do I need to use a tool? No
Final Answer: [Your helpful response here, or the JSON output from the tool if it was a data fetch]

IMPORTANT:
- If the user asks for data (meds, appointments), USE the tool.
- If the tool returns a JSON object with "action": "navigate" or "display", your Final Answer should be that JSON object directly (as a string).
- If the tool returns a message, incorporate it into your Final Answer.
- If the user asks a general question, use answer_health_question or just answer directly.

Context: {user_context}
User Input: {user_input}
"""
        
        # Simple Loop (User -> LLM -> Tool -> LLM -> Final)
        # We will do up to 3 iterations
        
        conversation_history = [{"role": "system", "content": system_prompt}]
        
        # Add history if provided
        if history:
            for msg in history:
                role = "assistant" if msg.get("role") == "ai" else msg.get("role", "user")
                content = msg.get("message", "") or msg.get("content", "")
                if content:
                    conversation_history.append({"role": role, "content": content})

        conversation_history.append({"role": "user", "content": user_input})
        
        pending_action = None # To store 'display' actions to return with final answer

        try:
            for i in range(3): # Max 3 steps
                # Step 1: LLM decides what to do
                # Add stop token to prevent hallucination of Observation
                try:
                    response = await cls._llm.ainvoke(conversation_history, stop=["Observation:"])
                except Exception as e:
                    # Check for rate limit
                    if "429" in str(e) or "Rate limit" in str(e):
                        logger.warning(f"Groq Rate Limit in Agent. Switching to OpenRouter fallback (HTTP). Error: {e}")
                        
                        # Direct HTTP call to OpenRouter to avoid openai dependency
                        try:
                            headers = {
                                "Authorization": f"Bearer {settings.openrouter_api_key}",
                                "Content-Type": "application/json",
                                "HTTP-Referer": "https://healthsync.ai",
                                "X-Title": "HealthSync AI"
                            }
                            
                            payload = {
                                "model": settings.openrouter_model,
                                "messages": conversation_history,
                                "temperature": 0.0
                            }
                            if "stop" in locals(): # stop kwarg from ainvoke
                                payload["stop"] = ["Observation:"]

                            async with httpx.AsyncClient() as client:
                                or_resp = await client.post(
                                    "https://openrouter.ai/api/v1/chat/completions",
                                    headers=headers,
                                    json=payload,
                                    timeout=30.0
                                )
                                or_resp.raise_for_status()
                                or_data = or_resp.json()
                                content = or_data["choices"][0]["message"]["content"]
                                
                                # Mock response object for consistency
                                class MockResponse:
                                    def __init__(self, c): self.content = c
                                response = MockResponse(content)
                                
                        except Exception as or_e:
                            logger.error(f"OpenRouter direct fallback failed: {or_e}")
                            raise e # Raise original if fallback fails
                    else:
                        raise e
                        
                content = response.content
                
                logger.info(f"Agent Step {i+1}: {content}")
                
                # Append assistant message to history
                conversation_history.append({"role": "assistant", "content": content})
                
                if "Action:" in content:
                    # Parse Action
                    action_match = re.search(r"Action:\s*(.*)", content)
                    input_match = re.search(r"Action Input:\s*(.*)", content)
                    
                    if action_match:
                        action_name = action_match.group(1).strip()
                        action_input = input_match.group(1).strip() if input_match else ""
                        
                        logger.info(f"Executing Tool: {action_name} with input: {action_input}")

                        # Execute Tool
                        tool_result = None
                        if action_name in cls._tools:
                            tool = cls._tools[action_name]
                            try:
                                # Check if tool is async
                                if tool.coroutine:
                                    tool_result = await tool.arun(action_input)
                                else:
                                    tool_result = tool.run(action_input)
                            except Exception as e:
                                tool_result = f"Error executing tool: {e}"
                        else:
                            tool_result = f"Error: Tool '{action_name}' not found."
                        
                        logger.info(f"Tool Result: {tool_result}")
                        
                        # Handling structured results
                        if isinstance(tool_result, dict):
                            action_type = tool_result.get("action")
                            
                            # If navigation, return immediately (highest priority, change context)
                            if action_type == "navigate":
                                return tool_result
                                
                            # If display, store it to attach to final answer, 
                            # BUT let the LLM see the data so it can answer specific questions about it.
                            if action_type == "display":
                                pending_action = tool_result
                                # Create a text summary for the LLM
                                data_summary = json.dumps(tool_result.get("data", "Data displayed on screen"))
                                observation = f"Observation: The data has been fetched and shown to the user. Data content: {data_summary}"
                                conversation_history.append({"role": "user", "content": observation})
                                continue

                            return tool_result
                        
                        # If string result (like from answer_health_question), feed back to LLM
                        observation = f"Observation: {tool_result}"
                        conversation_history.append({"role": "user", "content": observation})
                        continue # Loop again to get Final Answer
                
                # If no action, or we have a Final Answer
                if "Final Answer:" in content:
                    final_answer = content.split("Final Answer:")[-1].strip()
                    
                    # Try to parse if the final answer itself is a JSON action (hallucination catch)
                    try:
                        if final_answer.startswith('{') and final_answer.endswith('}'):
                            possible_json = json.loads(final_answer)
                            if "action" in possible_json:
                                # Start with this as the response object
                                response_obj = possible_json
                                # Ensure it has a message field if missing
                                if "message" not in response_obj:
                                    response_obj["message"] = "Processing your request..."
                                
                                # If we had a pending action (e.g. from a real tool), merge/override?
                                # Usually if LLM gives JSON final answer, it replaces the pending one or IS the result.
                                # Let's respect the pending action if it exists (real data) over hallucinated one,
                                # OR if they match, fine.
                                # But if pending_action exists, we probably want that one.
                                if pending_action:
                                     # Merge: start with pending_action, override message with this JSON's message if meaningful?
                                     # Actually, simple heuristic: Use pending_action if available.
                                     response_obj = pending_action.copy()
                                     # If the LLM JSON had a message, use it.
                                     if "message" in possible_json:
                                         response_obj["message"] = possible_json["message"]
                                
                                return response_obj
                    except:
                        pass

                    response_obj = {
                        "type": "message",
                        "message": final_answer,
                        "data": None
                    }
                    # Attach pending display action if exists
                    if pending_action:
                        # We merge them. 
                        response_obj = pending_action.copy()
                        response_obj["message"] = final_answer # Override the default tool message with LLM's answer
                    
                    return response_obj
                
                # If LLM didn't produce Action or Final Answer, it might be just chatting.
                # Treat as final answer.
                if not "Action:" in content:
                     return {
                        "type": "message",
                        "message": content,
                        "data": None
                    }

            # Fallback if loop exhausted
            return {
                "type": "error",
                "message": "I processed your request but couldn't finalize an answer."
            }

        except Exception as e:
            logger.error(f"Error in manual agent loop: {e}")
            return {
                "type": "error",
                "message": "Sorry, I had trouble processing that."
            }
