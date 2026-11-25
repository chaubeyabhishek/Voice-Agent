import os
import sys
import asyncio
import logging

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import config
from stt import transcribe_audio, get_whisper_model
from tts import text_to_speech, get_autoplay_html, select_appropriate_voice
from llm_chain import AdmissionsConversationManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run_e2e_tests():
    print("=" * 65)
    print("STARTING COMPLETE VOICE-TO-VOICE PIPELINE VALIDATION TEST")
    print("=" * 65)

    print("\n[Step 1 & 2] Initializing LangChain Memory and Faster-Whisper Model...")
    mgr = AdmissionsConversationManager()
    whisper_model = get_whisper_model()
    assert whisper_model is not None, "Whisper model failed to load!"
    print("✓ Faster-Whisper model and LangChain memory initialized.")

    test_cases = [
        {
            "lang_label": "English",
            "audio_prompt": "Hello Counselor! Can you tell me what B Tech engineering programs are offered?",
            "tts_voice": "en-IN-NeerjaNeural"
        },
        {
            "lang_label": "Hinglish",
            "audio_prompt": "Mera 12th complete ho gaya hai with 85 percent, kya mujhe scholarship mil sakti hai?",
            "tts_voice": "en-IN-NeerjaNeural"
        },
        {
            "lang_label": "Hindi",
            "audio_prompt": "नमस्ते, क्या मुझे कैंपस में हॉस्टल की सुविधा मिलेगी?",
            "tts_voice": "hi-IN-SwaraNeural"
        }
    ]

    for idx, case in enumerate(test_cases, start=1):
        print(f"\n--- [Turn {idx} - Testing {case['lang_label']}] ---")

        print(f"1. Simulating microphone capture of student speaking: '{case['audio_prompt']}'")
        student_audio_bytes = text_to_speech(case['audio_prompt'], case['tts_voice'])
        assert len(student_audio_bytes) > 0, "Failed to simulate student microphone audio!"
        print(f"✓ Microphone audio captured: {len(student_audio_bytes)} bytes.")

        print("2. Transcribing student audio using Faster-Whisper (STT)...")
        stt_result = transcribe_audio(student_audio_bytes)
        transcribed_text = stt_result.get("text", "")
        detected_lang = stt_result.get("language", "")
        print(f"✓ Transcribed text: '{transcribed_text}' [Detected Language: {detected_lang}]")
        assert len(transcribed_text) > 0, f"STT failed to transcribe {case['lang_label']} audio!"

        print("3. Passing transcribed text & conversation history to LangChain...")
        ai_response = mgr.generate_response(transcribed_text)
        print(f"✓ AI Response generated:\n   \"{ai_response}\"")
        assert len(ai_response) > 0, "LLM failed to generate a response!"

        print("4. Converting AI response to speech using Edge-TTS...")
        chosen_voice = select_appropriate_voice(ai_response, case['tts_voice'])
        ai_audio_bytes = text_to_speech(ai_response, chosen_voice)
        assert len(ai_audio_bytes) > 0, "Edge-TTS failed to generate audio!"
        print(f"✓ Edge-TTS synthesized speech: {len(ai_audio_bytes)} bytes (Voice: {chosen_voice}).")

        print("5. Generating safe auto-play audio component...")
        autoplay_html = get_autoplay_html(ai_audio_bytes, f"test_turn_{idx}")
        assert "audio/mp3;base64," in autoplay_html, "Autoplay HTML missing base64 audio!"
        assert "autoplay" in autoplay_html, "Autoplay attribute missing!"
        print("✓ Autoplay HTML ready for browser playback.")

    print("\n--- [Contextual Memory Test Turn] ---")
    memory_query = "What three questions did I ask you about earlier in our conversation?"
    print(f"Asking memory retention question: '{memory_query}'")
    memory_response = mgr.generate_response(memory_query)
    print(f"✓ AI Context Response:\n   \"{memory_response}\"")

    messages = mgr.get_messages()
    print(f"Total messages in LangChain memory: {len(messages)}")
    assert len(messages) >= 8, f"Expected at least 8 messages in memory, got {len(messages)}"
    print("✓ Contextual conversation memory verified across turns!")

    print("\n" + "=" * 65)
    print("ALL VOICE-TO-VOICE PIPELINE TESTS PASSED SUCCESSFULLY! (100% COMPLETE)")
    print("=" * 65)

if __name__ == "__main__":
    run_e2e_tests()
