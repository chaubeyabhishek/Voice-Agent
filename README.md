# 🎙️ Two-Way Voice AI Admissions Counselor

A real-time voice-to-voice interactive application that enables students to talk directly with an AI Admissions Counselor using their microphone and hear natural spoken audio responses.

---

## 🌟 Key Features

1. **Two-Way Voice Interaction**:
   - Speak naturally through the microphone.
   - Hear the AI Counselor's response spoken back automatically without clicking Play.
2. **Multilingual Support**:
   - Supports **English** (Indian and US accents), **हिंदी (Hindi)**, and **Hinglish** (bilingual mix).
3. **Speech-to-Text (STT)**:
   - Powered by **Faster-Whisper** with Voice Activity Detection (VAD) and automatic language identification.
4. **Contextual Memory & Reasoning**:
   - Powered by **LangChain** and **Google Gemini LLM** (`gemini-2.5-flash`).
   - Retains context across multiple turns of conversation.
5. **Text-to-Speech (TTS)**:
   - High-quality neural speech synthesis using **Edge-TTS** (`en-IN-NeerjaNeural`, `hi-IN-SwaraNeural`, etc.).
6. **Safety & Robustness**:
   - Strict guarantee: Audio is **never** played before TTS generation is fully complete.
   - Handles silent audio, microphone permissions, and API errors gracefully.
7. **Session Controls**:
   - **Start Conversation**: Initializes speech models and starts live session.
   - **Stop / Pause Conversation**: Pauses recording.
   - **End & Clear Conversation**: Resets memory and transcript.

---

## 🚀 How to Run the Application

Activate the virtual environment and start Streamlit:

```bash
# Run the Streamlit application
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open your browser at:
```
http://localhost:8501
```

---

## 🧪 Running the End-to-End Pipeline Tests

To run the complete automated validation suite (testing STT, LangChain Gemini, Edge-TTS, and conversation memory across English, Hinglish, and Hindi):

```bash
.\.venv\Scripts\python.exe test_pipeline.py
```

---

## 📁 Architecture Overview

- **`app.py`**: Streamlit user interface, microphone capture, status indicators, audio auto-play, and transcript display.
- **`stt.py`**: Faster-Whisper model loading and audio transcription.
- **`tts.py`**: Edge-TTS neural speech synthesis and auto-play HTML component generation.
- **`llm_chain.py`**: LangChain `InMemoryChatMessageHistory` and `ChatGoogleGenerativeAI` conversation manager.
- **`config.py`**: Configuration for models, voices, and voice-optimized system prompts.
- **`test_pipeline.py`**: End-to-end validation test suite.
