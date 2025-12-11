# autopodcast/config.py

from dataclasses import dataclass, field
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

@dataclass
class WhisperConfig:
    model_size: str = "medium"   # "small", "medium", "large"
    device: str = "cuda"         # or "cpu"
    language: str | None = None  # let Whisper auto-detect if None

@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"  # or OpenAI embedding model id
    dim: int = 384

@dataclass
class SegmentationConfig:
    window_seconds: int = 60
    min_chapter_length_seconds: int = 120
    similarity_threshold: float = 0.50

@dataclass
class SummarizationConfig:
    max_tokens: int = 256
    style: str = "concise"
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))

@dataclass
class SearchConfig:
    top_k: int = 5

@dataclass
class AppConfig:
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

CONFIG = AppConfig()
