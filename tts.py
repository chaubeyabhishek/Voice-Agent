import asyncio
import base64
import os
import re
import tempfile
import logging
from typing import Optional
import edge_tts
import config

logger = logging.getLogger(__name__)

def is_hindi_script(text: str) -> bool:
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    return bool(devanagari_pattern.search(text))

def select_appropriate_voice(text: str, user_selected_voice: Optional[str] = None) -> str:
    if user_selected_voice:
        return user_selected_voice
    if is_hindi_script(text):
        return "hi-IN-SwaraNeural"
    return config.DEFAULT_VOICE

async def _synthesize_async(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    audio_buffer = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.extend(chunk["data"])
    return bytes(audio_buffer)

def text_to_speech(
    text: str,
    voice: Optional[str] = None
) -> bytes:
    if not text or not text.strip():
        return b""

    cleaned_text = re.sub(r'[\*\#\_\[\]\(\)\<\>]', '', text).strip()
    target_voice = select_appropriate_voice(cleaned_text, voice)

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(_synthesize_async(cleaned_text, target_voice))
            else:
                return loop.run_until_complete(_synthesize_async(cleaned_text, target_voice))
        except RuntimeError:
            return asyncio.run(_synthesize_async(cleaned_text, target_voice))

    except Exception as e:
        logger.error(f"TTS synthesis error: {e}", exc_info=True)
        return b""

def get_autoplay_html(audio_bytes: bytes, audio_id: str = "ai_voice") -> str:
    # Critical requirement: only rendered after audio_bytes are completely generated
    if not audio_bytes:
        return ""

    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    audio_element_id = f"audio_{audio_id}"

    # Dark-mode styled audio player
    html_code = f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-left: 4px solid #6366f1; padding: 12px 16px; border-radius: 10px; margin: 10px 0; backdrop-filter: blur(8px);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 0.85rem; font-weight: 600; color: #818cf8;">🔊 AI Voice Response (Auto-Playing)</span>
            <span style="font-size: 0.75rem; color: #94a3b8; background: #0f172a; padding: 2px 8px; border-radius: 4px; border: 1px solid #1e293b;">Live Audio</span>
        </div>
        <audio id="{audio_element_id}" controls autoplay playsinline style="width: 100%; height: 38px; filter: invert(0.88) hue-rotate(180deg);">
            <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
            Your browser does not support audio playback.
        </audio>
    </div>
    <script>
        (function() {{
            const el = document.getElementById("{audio_element_id}");
            if (el) {{
                el.currentTime = 0;
                const playPromise = el.play();
                if (playPromise !== undefined) {{
                    playPromise.catch(function(err) {{
                        console.warn("Autoplay notice:", err);
                    }});
                }}
            }}
        }})();
    </script>
    """
    return html_code
