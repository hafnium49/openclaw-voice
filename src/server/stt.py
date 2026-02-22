"""
Speech-to-Text module using Whisper or Deepgram.
"""

import asyncio
import io
import json
import urllib.request
import wave
from typing import Optional

import numpy as np
from loguru import logger


class WhisperSTT:
    """Whisper-based Speech-to-Text."""
    
    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        language: str = "en",
        provider: str = "whisper",
        deepgram_api_key: Optional[str] = None,
        deepgram_model: str = "nova-3",
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.provider = provider
        self.deepgram_api_key = deepgram_api_key
        self.deepgram_model = deepgram_model
        self.model = None
        self._backend = "mock"
        self._load_model()
    
    def _load_model(self):
        """Load STT backend."""
        if self.provider == "deepgram":
            if not self.deepgram_api_key:
                logger.warning("Deepgram provider selected but API key missing; falling back")
            else:
                self._backend = "deepgram"
                logger.info(f"✅ Deepgram STT ready (model={self.deepgram_model}, lang={self.language})")
                return

        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel
            
            if self.device == "auto":
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    compute_type = "float16"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "cpu"
                    compute_type = "int8"
                else:
                    self.device = "cpu"
                    compute_type = "int8"
            elif self.device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"
            
            logger.info(f"Loading faster-whisper {self.model_name} on {self.device}")
            self.model = WhisperModel(
                self.model_name,
                device=self.device if self.device != "mps" else "cpu",
                compute_type=compute_type,
            )
            self._backend = "faster-whisper"
            logger.info("✅ faster-whisper loaded")
            return
        except ImportError:
            logger.warning("faster-whisper not available")
        except Exception as e:
            logger.warning(f"faster-whisper failed: {e}")
        
        # Try openai-whisper
        try:
            import whisper
            
            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading openai-whisper {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            self._backend = "openai-whisper"
            logger.info("✅ openai-whisper loaded")
            return
        except ImportError:
            logger.warning("openai-whisper not available")
        except Exception as e:
            logger.warning(f"openai-whisper failed: {e}")
        
        # Mock mode for testing
        logger.warning("⚠️ No STT backend - using mock mode")
        self._backend = "mock"
    
    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)
    
    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription."""
        if self._backend == "deepgram":
            return self._transcribe_deepgram(audio)

        if self._backend == "faster-whisper":
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(segment.text for segment in segments).strip()
        
        elif self._backend == "openai-whisper":
            result = self.model.transcribe(audio, language=self.language)
            return result["text"].strip()
        
        else:
            # Mock mode - return placeholder
            logger.debug(f"Mock STT: received {len(audio)} samples")
            return "[Mock transcription - install whisper for real STT]"

    def _transcribe_deepgram(self, audio: np.ndarray) -> str:
        """Transcribe using Deepgram prerecorded REST endpoint."""
        try:
            # Ensure float32 mono in [-1, 1]
            audio = np.asarray(audio, dtype=np.float32)
            audio = np.clip(audio, -1.0, 1.0)
            pcm16 = (audio * 32767.0).astype(np.int16)

            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(pcm16.tobytes())

            url = (
                f"https://api.deepgram.com/v1/listen?model={self.deepgram_model}"
                f"&language={self.language}&smart_format=true&punctuate=true"
            )
            req = urllib.request.Request(
                url,
                data=wav_buf.getvalue(),
                headers={
                    "Authorization": f"Token {self.deepgram_api_key}",
                    "Content-Type": "audio/wav",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))

            return (
                payload.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
                .strip()
            )
        except Exception as e:
            logger.error(f"Deepgram STT error: {e}")
            return ""
