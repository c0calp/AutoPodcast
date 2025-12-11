# autopodcast/pipeline.py

from __future__ import annotations

from .models import AudioFileInfo, PodcastIndex
from .audio_preprocessing import load_audio
from .asr_whisper import WhisperTranscriber
from .cleaning import clean_transcript
from .topic_segmentation import segment_into_chapters
from .summarization import summarize_chapters
from .keywords import assign_keywords_to_chapters, build_global_keywords
from .search import SemanticSearchIndex


def process_podcast(audio_path: str) -> tuple[PodcastIndex, SemanticSearchIndex]:
    # 1. Load / convert audio
    audio_info: AudioFileInfo = load_audio(audio_path)

    # 2. Transcribe
    transcriber = WhisperTranscriber()
    raw_transcript = transcriber.transcribe(audio_info)

    # 3. Clean transcript
    transcript = clean_transcript(raw_transcript)

    # 4. Topic segmentation → chapters
    chapters = segment_into_chapters(transcript)

    # 5. Summaries
    chapters = summarize_chapters(chapters)

    # 6. Keywords
    chapters = assign_keywords_to_chapters(chapters)
    global_keywords = build_global_keywords(chapters)

    # 7. Search index
    search_index = SemanticSearchIndex()
    search_index.build(transcript)

    index = PodcastIndex(
        audio=audio_info,
        transcript=transcript,
        chapters=chapters,
        global_keywords=global_keywords,
    )

    return index, search_index
