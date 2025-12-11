# autopodcast/cleaning.py

import re
from .models import Transcript, TranscriptSegment

FILLER_WORDS = [
    "uh", "umm", "um", "you know", "like", "sort of", "kind of"
]

def clean_segment_text(text: str) -> str:
    """
    Remove filler words, normalize whitespace, simple punctuation fixes.
    """
    # TODO: more sophisticated cleaning if needed
    t = text

    # Remove filler words (very naive)
    for fw in FILLER_WORDS:
        pattern = r"\b" + re.escape(fw) + r"\b"
        t = re.sub(pattern, " ", t, flags=re.IGNORECASE)

    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_transcript(transcript: Transcript) -> Transcript:
    """
    Apply cleaning to each TranscriptSegment.
    """
    cleaned_segments: list[TranscriptSegment] = []
    for seg in transcript.segments:
        cleaned_text = clean_segment_text(seg.text)
        if not cleaned_text:
            continue
        cleaned_segments.append(
            TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=cleaned_text,
                speaker=seg.speaker,
                confidence=seg.confidence
            )
        )
    return Transcript(segments=cleaned_segments)
