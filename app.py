import os
import sys
import time
import base64
import logging
from datetime import datetime

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import streamlit as st
from streamlit_mic_recorder import mic_recorder

import config
from stt import transcribe_audio, get_whisper_model
from tts import text_to_speech, get_autoplay_html, select_appropriate_voice
from llm_chain import AdmissionsConversationManager

st.set_page_config(
    page_title="AI Assistant Admissions Counselor",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode CSS theme and responsive card styling
st.markdown("""
<style>
    /* Dark theme root overrides */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 950px;
    }

    /* Active call badge with glowing pulse dot */
    .call-active-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.35);
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.15);
    }
    .call-pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse-dark 1.6s infinite;
    }
    @keyframes pulse-dark {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.75); }
        70% { transform: scale(1.15); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Idle status badge */
    .call-inactive-badge {
        display: inline-flex;
        align-items: center;
        background: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 500;
        font-size: 0.85rem;
    }

    /* Dark voice recording card */
    .voice-control-box {
        background: linear-gradient(145deg, #0f172a 0%, #131d33 100%);
        border: 2px dashed #334155;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    /* Dark start hero card */
    .hero-card {
        background: linear-gradient(145deg, #111c30 0%, #0d1527 100%);
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        box-shadow: 0 12px 30px -4px rgba(0, 0, 0, 0.5);
        margin-bottom: 25px;
    }

    /* Timestamp & badges */
    .chat-timestamp {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 5px;
    }
    .lang-badge {
        font-size: 0.7rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.25);
        color: #c7d2fe;
        border: 1px solid rgba(165, 180, 252, 0.35);
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 6px;
    }

    /* Audio player dark styling */
    audio {
        filter: invert(0.88) hue-rotate(180deg);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

if "conversation_active" not in st.session_state:
    st.session_state.conversation_active = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation_manager" not in st.session_state:
    st.session_state.conversation_manager = AdmissionsConversationManager()

if "last_processed_audio_id" not in st.session_state:
    st.session_state.last_processed_audio_id = None

if "autoplay_latest_audio" not in st.session_state:
    st.session_state.autoplay_latest_audio = True

if "selected_voice" not in st.session_state:
    st.session_state.selected_voice = config.DEFAULT_VOICE

if "latest_audio_bytes" not in st.session_state:
    st.session_state.latest_audio_bytes = None

if "latest_audio_id" not in st.session_state:
    st.session_state.latest_audio_id = None

with st.sidebar:
    st.title("🎙️ Voice Settings")

    st.subheader("🔑 Gemini API")
    env_key = os.getenv("GEMINI_API_KEY", "")
    user_api_key = st.text_input(
        "Gemini API Key",
        value=env_key,
        type="password",
        help="Google Gemini API Key for LangChain LLM reasoning"
    )
    if user_api_key != env_key and user_api_key:
        st.session_state.conversation_manager.update_api_key(user_api_key)
        st.success("API Key updated!")

    st.divider()

    st.subheader("🗣️ Voice & Accent")
    voice_display_names = list(config.VOICE_OPTIONS.keys())
    current_voice_index = 0
    for idx, (name, voice_code) in enumerate(config.VOICE_OPTIONS.items()):
        if voice_code == st.session_state.selected_voice:
            current_voice_index = idx
            break

    selected_voice_display = st.selectbox(
        "Counselor Voice",
        options=voice_display_names,
        index=current_voice_index,
        help="Select the voice persona for Edge-TTS audio output"
    )
    st.session_state.selected_voice = config.VOICE_OPTIONS[selected_voice_display]

    st.session_state.autoplay_latest_audio = st.toggle(
        "Auto-Play Voice Responses",
        value=st.session_state.autoplay_latest_audio,
        help="Automatically plays the AI counselor's voice without clicking Play"
    )

    st.divider()

    st.subheader("🌐 Supported Languages")
    st.markdown("""
    - **English** (Native / Indian Accent)
    - **हिंदी** (Hindi conversational)
    - **Hinglish** (Bilingual mix, e.g. *"Courses ke baare mein bataiye"*)
    """)

    st.divider()

    st.subheader("📊 Session Stats")
    total_turns = len(st.session_state.chat_history)
    st.metric("Total Turns", total_turns)
    if total_turns > 0:
        if st.button("🗑️ Reset Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.conversation_manager.clear_memory()
            st.session_state.last_processed_audio_id = None
            st.session_state.latest_audio_bytes = None
            st.rerun()

col_title, col_status = st.columns([0.68, 0.32], vertical_alignment="center")

with col_title:
    st.markdown("### 🎓 University Admissions Voice Agent")
    st.caption("Real-Time Voice-to-Voice AI Counselor • Whisper STT + LangChain Gemini + Edge-TTS")

with col_status:
    if st.session_state.conversation_active:
        st.markdown("""
        <div style="text-align: right;">
            <div class="call-active-badge">
                <span class="call-pulse-dot"></span> Call Connected
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: right;">
            <div class="call-inactive-badge">
                ⏹️ Session Idle
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

if not st.session_state.conversation_active:
    st.markdown("""
    <div class="hero-card">
        <h2 style="color: #f8fafc; margin-bottom: 8px;">Start Your Voice Consultation</h2>
        <p style="color: #94a3b8; font-size: 1.02rem; max-width: 650px; margin: 0 auto 20px auto;">
            Speak naturally with Counselor Aarav about admissions, eligibility, courses, fees, or scholarships. 
            Speak in <b style="color: #f1f5f9;">English</b>, <b style="color: #f1f5f9;">हिंदी (Hindi)</b>, or <b style="color: #f1f5f9;">Hinglish</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🟢 Start Conversation", type="primary", use_container_width=True):
            st.session_state.conversation_active = True
            with st.spinner("Connecting voice channels and initializing speech engine..."):
                get_whisper_model()
            st.rerun()

else:
    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        if st.button("⏸️ Stop / Pause Conversation", use_container_width=True):
            st.session_state.conversation_active = False
            st.info("Conversation paused. Click Start Conversation to resume anytime.")
            st.rerun()

    with col_ctrl2:
        if st.button("❌ End & Clear Conversation", type="secondary", use_container_width=True):
            st.session_state.conversation_active = False
            st.session_state.chat_history = []
            st.session_state.conversation_manager.clear_memory()
            st.session_state.last_processed_audio_id = None
            st.session_state.latest_audio_bytes = None
            st.rerun()

if st.session_state.conversation_active:
    st.markdown("""
    <div class="voice-control-box">
        <div style="font-weight: 600; color: #f8fafc; font-size: 1.05rem; margin-bottom: 4px;">
            🎙️ Speak to Your Admissions Counselor
        </div>
        <div style="color: #94a3b8; font-size: 0.88rem; margin-bottom: 12px;">
            Click <b>Start Recording</b>, speak your question, then click <b>Stop & Send</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    audio_record = mic_recorder(
        start_prompt="🔴 Start Recording (Speak Now)",
        stop_prompt="⏹️ Stop & Send Audio",
        just_once=False,
        use_container_width=True,
        key="admissions_mic_recorder",
        format="webm"
    )

    if audio_record is not None and "bytes" in audio_record:
        audio_id = audio_record.get("id")

        # Debounce: prevent duplicate processing of the same recording ID across reruns
        if audio_id != st.session_state.last_processed_audio_id:
            st.session_state.last_processed_audio_id = audio_id
            audio_bytes = audio_record["bytes"]

            if len(audio_bytes) < 800:
                st.warning("⚠️ Recording was too short or silent. Please speak into your microphone and try again.")
            else:
                with st.status("Processing your voice...", expanded=True) as status_box:
                    st.write("⚙️ **Step 1/3:** Transcribing voice using Faster-Whisper...")
                    transcription_res = transcribe_audio(audio_bytes)
                    student_text = transcription_res.get("text", "").strip()
                    detected_lang = transcription_res.get("language", "en")

                    if not student_text:
                        status_box.update(label="No clear speech detected", state="error", expanded=False)
                        st.warning("Could not detect any clear speech. Please ensure your microphone is working and speak clearly.")
                    else:
                        st.write(f"✓ Transcribed [{detected_lang.upper()}]: *\"{student_text}\"*")

                        st.write("🤖 **Step 2/3:** LangChain & Gemini generating contextual response...")
                        ai_response_text = st.session_state.conversation_manager.generate_response(student_text)
                        st.write("✓ Response generated!")

                        st.write("🔊 **Step 3/3:** Edge-TTS converting response to neural speech...")
                        chosen_voice = select_appropriate_voice(ai_response_text, st.session_state.selected_voice)
                        ai_audio_bytes = text_to_speech(ai_response_text, chosen_voice)
                        st.write("✓ Speech synthesis complete!")

                        status_box.update(label="Completed! Playing response...", state="complete", expanded=False)

                        current_time = datetime.now().strftime("%I:%M %p")

                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": student_text,
                            "language": detected_lang,
                            "timestamp": current_time,
                        })

                        turn_audio_id = f"turn_{int(time.time())}"

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": ai_response_text,
                            "audio_bytes": ai_audio_bytes,
                            "voice": chosen_voice,
                            "audio_id": turn_audio_id,
                            "timestamp": current_time,
                        })

                        st.session_state.latest_audio_bytes = ai_audio_bytes
                        st.session_state.latest_audio_id = turn_audio_id
                        st.rerun()

# Critical requirement: Autoplay is only injected after TTS generation is completely verified
if st.session_state.conversation_active and st.session_state.autoplay_latest_audio:
    if st.session_state.latest_audio_bytes and len(st.session_state.latest_audio_bytes) > 0:
        autoplay_code = get_autoplay_html(
            st.session_state.latest_audio_bytes,
            st.session_state.latest_audio_id or "latest"
        )
        st.markdown(autoplay_code, unsafe_allow_html=True)
        st.session_state.latest_audio_bytes = None

st.markdown("---")
st.subheader("💬 Conversation Transcript")

if not st.session_state.chat_history:
    if st.session_state.conversation_active:
        st.info("🎙️ Conversation started! Press the recording button above and speak your question.")
    else:
        st.markdown(
            "<div style='color: #64748b; text-align: center; padding: 25px;'>"
            "No messages yet. Click <b>Start Conversation</b> above to begin speaking with Counselor Aarav."
            "</div>",
            unsafe_allow_html=True
        )
else:
    for idx, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🎓"):
                lang_tag = f"<span class='lang-badge'>{msg.get('language', 'en').upper()}</span>"
                st.markdown(f"**Student** {lang_tag}", unsafe_allow_html=True)
                st.write(msg["content"])
                st.markdown(f"<div class='chat-timestamp'>{msg.get('timestamp', '')}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("**Counselor Aarav (AI)**", unsafe_allow_html=True)
                st.write(msg["content"])
                if msg.get("audio_bytes"):
                    st.audio(msg["audio_bytes"], format="audio/mp3")
                st.markdown(f"<div class='chat-timestamp'>{msg.get('timestamp', '')}</div>", unsafe_allow_html=True)
