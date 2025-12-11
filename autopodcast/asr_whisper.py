# autopodcast/asr_whisper.py

from __future__ import annotations
from typing import List

from .config import CONFIG
from .models import AudioFileInfo, Transcript, TranscriptSegment

try:
    import whisper
except ImportError:
    whisper = None


class WhisperTranscriber:
    def __init__(self):
        self.cfg = CONFIG.whisper
        self._model = None

    def load_model(self):
        """
        Lazily load the Whisper model.
        """
        if whisper is None:
            raise ImportError(
                "whisper package not installed. "
                "Install with: pip install -U openai-whisper"
            )

        if self._model is None:
            self._model = whisper.load_model(self.cfg.model_size)

    def transcribe(self, audio: AudioFileInfo) -> Transcript:
        """
        Run Whisper and convert the result into Transcript/TranscriptSegment objects.
        """
        self.load_model()

        result = self._model.transcribe(
            audio.path,
            language=self.cfg.language,
            verbose=False,
        )

        segments: List[TranscriptSegment] = []
        for seg in result.get("segments", []):
            segments.append(
                TranscriptSegment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=seg["text"].strip(),
                    speaker=None,
                    confidence=float(seg.get("avg_logprob", 0.0)),
                )
            )

        return Transcript(segments=segments)
