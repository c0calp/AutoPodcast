# autopodcast/topic_segmentation.py

from __future__ import annotations
from typing import List
import numpy as np
from dataclasses import dataclass
import json
import os

from .models import Transcript, TranscriptSegment, Chapter
from .embeddings import EmbeddingModel
from .config import CONFIG
import google.generativeai as genai

# Configure Google GenAI API
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)


@dataclass
class SegmentedTranscriptWindow:
    start: float
    end: float
    text: str
    segments: List[TranscriptSegment]


def window_transcript(transcript: Transcript) -> List[SegmentedTranscriptWindow]:
    """
    Group segments into fixed time windows (e.g., 60 seconds).
    """
    cfg = CONFIG.segmentation
    windows: List[SegmentedTranscriptWindow] = []
    segs = transcript.segments
    if not segs:
        return windows

    current_segments: List[TranscriptSegment] = []
    current_start = segs[0].start
    current_end = current_start + cfg.window_seconds

    for seg in segs:
        if seg.start < current_end:
            current_segments.append(seg)
        else:
            # flush
            if current_segments:
                text = " ".join(s.text for s in current_segments)
                windows.append(
                    SegmentedTranscriptWindow(
                        start=current_start,
                        end=current_end,
                        text=text,
                        segments=current_segments,
                    )
                )
            # new window
            current_segments = [seg]
            current_start = seg.start
            current_end = current_start + cfg.window_seconds

    if current_segments:
        text = " ".join(s.text for s in current_segments)
        windows.append(
            SegmentedTranscriptWindow(
                start=current_start,
                end=current_end,
                text=text,
                segments=current_segments,
            )
        )

    return windows


def detect_topic_boundaries(
    windows: List[SegmentedTranscriptWindow],
    embeddings: np.ndarray,
) -> List[int]:
    """
    Return indices where a new chapter should start (0-based window index).

    Strategy:
    - Always start a chapter at window 0.
    - For each pair of adjacent windows, compute cosine similarity.
    - If similarity < cfg.similarity_threshold → start a new chapter at i.
    """
    cfg = CONFIG.segmentation


    if len(windows) == 0:
        return []
    if len(windows) == 1:
        return [0]

    chapter_starts = [0]

    # cosine similarity between adjacent windows
    for i in range(1, len(windows)):
        v1 = embeddings[i - 1]
        v2 = embeddings[i]

        denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

        if denom == 0.0:
            # fallback: treat as a hard boundary if one of the vectors is zero
            sim = -1.0
        else:
            sim = float(np.dot(v1, v2) / denom)

        if sim < cfg.similarity_threshold:
            chapter_starts.append(i)

    # ensure sorted & unique
    chapter_starts = sorted(set(chapter_starts))

    return chapter_starts


def build_chapters(
    windows: List[SegmentedTranscriptWindow],
    chapter_starts: List[int],
) -> List[Chapter]:
    """
    Merge windows into Chapter objects (no summaries yet).
    """
    cfg = CONFIG.segmentation
    chapters: List[Chapter] = []
    chapter_id = 0

    if not windows:
        return chapters

    chapter_starts = sorted(chapter_starts)
    for idx, start_idx in enumerate(chapter_starts):
        end_idx = chapter_starts[idx + 1] if idx + 1 < len(chapter_starts) else len(windows)

        chapter_windows = windows[start_idx:end_idx]
        segments = [s for w in chapter_windows for s in w.segments]
        if not segments:
            continue

        chapter_start = segments[0].start
        chapter_end = segments[-1].end

        # Merge too-short chapters into previous one
        # if (
        #     chapter_end - chapter_start < cfg.min_chapter_length_seconds
        #     and len(chapters) > 0
        # ):
        #     chapters[-1].segments.extend(segments)
        #     chapters[-1].end = chapter_end
        #     continue

        chapters.append(
            Chapter(
                id=chapter_id,
                start=chapter_start,
                end=chapter_end,
                summary="",
                keywords=[],
                segments=segments,
            )
        )
        chapter_id += 1

    return chapters


def segment_into_chapters(transcript: Transcript) -> List[Chapter]:
    """
    High-level: windows + embeddings + boundaries → Chapter objects (no summaries).
    """
    windows = window_transcript(transcript)
    texts = [w.text for w in windows]

    embedder = EmbeddingModel()
    embeddings = embedder.embed_texts(texts)  # shape (n, dim)

    chapter_starts = detect_topic_boundaries(windows, embeddings)
    chapters = build_chapters(windows, chapter_starts)
    return chapters


def enhance_chapters_with_genai(chapters: List[Chapter]) -> List[Chapter]:
    """
    Use Google Generative AI to generate intelligent chapter titles and descriptions.

    Args:
        chapters: List of Chapter objects with segments but potentially empty titles/summaries

    Returns:
        Updated Chapter objects with GenAI-generated titles and descriptions
    """
    if not GENAI_API_KEY:
        print("WARNING: GOOGLE_API_KEY not set. Skipping GenAI enhancement.")
        return chapters

    if not chapters:
        return chapters

    try:
        model = genai.GenerativeModel("gemini-pro")

        for i, chapter in enumerate(chapters):
            # Extract text from segments in this chapter
            chapter_text = " ".join([s.text for s in chapter.segments])

            if not chapter_text.strip():
                chapter.summary = f"Chapter {i+1}"
                continue

            # Limit text to avoid token limits (roughly ~2000 chars = ~500 tokens)
            chapter_text = chapter_text[:2000]

            prompt = f"""Analyze this podcast chapter excerpt and provide:
                1. A short, engaging chapter title (2-5 words)
                2. A brief description (1-2 sentences)

                Respond ONLY with valid JSON in this format:
                {{
                "title": "Chapter Title",
                "description": "Brief description of the chapter content"
                }}

                Chapter text:
                {chapter_text}

                Return only the JSON object, no other text."""

            response = model.generate_content(prompt)
            response_text = response.text.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]

            chapter_data = json.loads(response_text)
            chapter.summary = chapter_data.get("title", f"Chapter {i+1}")
            # Store description in keywords as a temporary location
            if "description" in chapter_data:
                chapter.keywords = [chapter_data["description"]]

            print(f"✓ Enhanced chapter {i+1}: {chapter.summary}")

    except Exception as e:
        print(f"Error enhancing chapters with GenAI: {e}")
        # Fallback: use default titles
        for i, chapter in enumerate(chapters):
            if not chapter.summary:
                chapter.summary = f"Chapter {i+1}"

    return chapters