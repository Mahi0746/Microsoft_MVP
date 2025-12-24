# HealthSync AI - System Workflow

This document outlines the end-to-end technical workflows for the HealthSync AI platform.

## 1. Authentication Workflow
1.  **Request**: User submits credentials via `/api/auth/signup` or `/api/auth/login`.
2.  **Processing**: 
    - Password is hashed/verified.
    - User record is retrieved from **MongoDB Atlas**.
    - **JWT Tokens** (Access & Refresh) are generated.
3.  **Response**: Tokens are sent back via **HTTP-only Cookies** for secure session management.
4.  **Verification**: Subsequent requests use the `get_current_user` dependency to validate cookies and authorize access.

## 2. AI Therapy Game (Physio-RPG) Workflow
The Therapy Game uses a **LangGraph-based AI agent** (the "Dungeon Master") to adapt the experience to the user.

### Mode Selection
- **Story Mode**: Generates a rich, 2-sentence narrative to motivate the user.
- **Express Mode**: Bypasses the storyteller to provide exercises 50% faster, ideal for busy or elderly users.

### Execution Flow
1.  **Frontend**: User selects a quest (e.g., Shoulder Rehab) and toggles Express Mode.
2.  **State Initialization**: `DungeonMasterService` initializes the game state with `difficulty`, `fatigue`, and `express_mode` flags.
3.  **Graph Logic**:
    - **Router**: Directs flow to `storyteller` (Story Mode) or jumps to `motion_architect` (Express Mode).
    - **Storyteller**: Generates a motivating scenario.
    - **Motion Architect**: Translates the story (or game type) into a concrete physical exercise with sets/reps.
    - **Safety Critic**: Validates the exercise for physical safety.
4.  **Exercise Execution**: User follows the exercise on-screen with real-time AI guidance.
5.  **Completion**: User reports performance; `performance_evaluator` updates progress and difficulty for the next turn.

## 3. AR Medical Scanner Workflow
1.  **Capture**: User points the camera at a medical document or physical injury.
2.  **Analysis**:
    - **OCR Service**: Extracts text from scanned documents.
    - **AR Scanner Service**: Processes visual data in real-time to identify medical markers.
3.  **Insights**: AI provides a summary of findings and recommends next steps in the dashboard.

## 4. Voice AI Health Assistant Workflow
1.  **Input**: User speaks symptoms or questions into the microphone.
2.  **Processing**:
    - **STT (Speech-to-Text)**: Transcribes user audio.
    - **AI Engine (Groq/LLM)**: Analyzes symptoms and context.
    - **TTS (Text-to-Speech)**: Responds with a natural AI voice.
3.  **Persistence**: Conversation history is stored in MongoDB for future context.

## 5. Google Fit Integration (Bio-Age Calculator)
1.  **Auth**: User authorizes Google Fit via OAuth2.
2.  **Data Pull**: System fetches real health metrics (Steps, Heart Rate, Sleep, Weight).
3.  **Bio-Age Logic**: 
    - Metrics are compared against chronological age.
    - System calculates a "Biological Age" and "Health Score" (0-100).
4.  **Feedback**: User receives personalized tips to lower their biological age based on gaps in their metrics.

## 6. Data Architecture
- **FastAPI**: Serves as the central orchestrator (API Gateway).
- **MongoDB Atlas**: Stores user profiles, health records, voice history, and game progress.
- **Local Storage**: Handles temporary uploads and caches.

---
*Created by the HealthSync AI Team*
