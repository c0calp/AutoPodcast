# autopodcast/models.py

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class AudioFileInfo:
    path: str
    duration_seconds: float | None = None
    sample_rate: int | None = None

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None

@dataclass
class Transcript:
    segments: List[TranscriptSegment] = field(default_factory=list)

    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)

@dataclass
class Chapter:
    id: int
    start: float
    end: float
    summary: str
    keywords: List[str]
    segments: List[TranscriptSegment] = field(default_factory=list)

@dataclass
class PodcastIndex:
    audio: AudioFileInfo
    transcript: Transcript
    chapters: List[Chapter]
    global_keywords: List[str]

@dataclass
class SearchResult:
    chapter_id: int
    segment_start: float
    segment_end: float
    snippet: str
    score: float
