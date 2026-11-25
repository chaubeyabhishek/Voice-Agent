import io
import os
import tempfile
import logging
from typing import Dict, Any, Union

# Workaround: PyAV 19 removed the deprecated 'metadata_errors' parameter expected by faster-whisper
import av
_orig_open = av.open
def _safe_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _orig_open(*args, **kwargs)
av.open = _safe_av_open

from faster_whisper import WhisperModel
import config

logger = logging.getLogger(__name__)

_whisper_instance = None

def get_whisper_model() -> WhisperModel:
    global _whisper_instance
    if _whisper_instance is None:
        download_root = config.WHISPER_DOWNLOAD_ROOT
        os.makedirs(download_root, exist_ok=True)
        _whisper_instance = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            download_root=download_root,
        )
    return _whisper_instance

def transcribe_audio(
    audio_input: Union[bytes, io.BytesIO, str],
    language: str = None
) -> Dict[str, Any]:
    if audio_input is None:
        return {"text": "", "language": "en", "probability": 0.0, "duration": 0.0}

    temp_path = None
    try:
        if isinstance(audio_input, (bytes, io.BytesIO)):
            raw_bytes = audio_input.getvalue() if isinstance(audio_input, io.BytesIO) else audio_input

            # Skip audio packets smaller than 1KB (empty silence or header only)
            if len(raw_bytes) < 1000:
                return {"text": "", "language": "en", "probability": 0.0, "duration": 0.0}

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(raw_bytes)
                temp_path = f.name
            audio_source = temp_path
        else:
            audio_source = audio_input

        model = get_whisper_model()

        segments, info = model.transcribe(
            audio_source,
            beam_size=5,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt="A bilingual Hindi and English conversation with students about college admissions."
        )

        segment_list = list(segments)
        transcribed_text = " ".join([seg.text.strip() for seg in segment_list]).strip()

        return {
            "text": transcribed_text,
            "language": info.language,
            "probability": getattr(info, "language_probability", 1.0),
            "duration": getattr(info, "duration", 0.0)
        }

    except Exception as e:
        logger.error(f"Error during audio transcription: {e}", exc_info=True)
        return {
            "text": "",
            "language": "en",
            "probability": 0.0,
            "duration": 0.0,
            "error": str(e)
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
