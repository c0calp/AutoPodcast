# ui/app_streamlit.py
from __future__ import annotations  # <-- must be first (after any docstring)

import os
import sys
from pathlib import Path

# from __future__ import annotations
import tempfile
from pathlib import Path

FFMPEG_DIR = r"/opt/homebrew/bin/ffmpeg"
os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

# Now the rest of your existing imports / path hack:
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st


from autopodcast.pipeline import process_podcast
from autopodcast.models import PodcastIndex
from autopodcast.search import SemanticSearchIndex


def render_audio_player(audio_path: str):
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/wav")


def render_chapters(index: PodcastIndex):
    st.header("Chapters")
    for ch in index.chapters:
        label = f"Chapter {ch.id} – {ch.start:.0f}s to {ch.end:.0f}s"
        with st.expander(label, expanded=(ch.id == 0)):
            st.markdown(f"**Summary**")
            st.write(ch.summary or "_(no summary)_")
            st.markdown(f"**Keywords:** {', '.join(ch.keywords) if ch.keywords else '–'}")
            st.markdown(f"**Time range:** `{ch.start:.1f}s – {ch.end:.1f}s`")


def render_search(search_index: SemanticSearchIndex, index: PodcastIndex):
    st.header("Semantic Search")
    query = st.text_input("Search in this episode", placeholder="e.g. AI safety, pricing, guest background...")
    if not query:
        return

    results = search_index.search(query)
    if not results:
        st.info("No results found.")
        return

    for r in results:
        st.markdown("---")
        st.markdown(f"⏱ **{r.segment_start:.1f}s – {r.segment_end:.1f}s**  (score: {r.score:.3f})")
        st.write(r.snippet)


def main():
    st.set_page_config(page_title="AutoPodcast", layout="wide")
    st.title("AutoPodcast – Podcast Indexing & Summarization")

    uploaded = st.file_uploader("Upload a podcast audio file", type=["mp3", "wav"])
    if not uploaded:
        st.info("Upload an audio file to start.")
        return

    # Save uploaded file to a temp path
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / uploaded.name
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        st.success("File uploaded. Processing…")

        # Run pipeline
        index, search_index = process_podcast(str(tmp_path))

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Audio")
            render_audio_player(index.audio.path)
            st.subheader("Global Tags")
            st.write(", ".join(index.global_keywords) if index.global_keywords else "–")

        with col2:
            render_chapters(index)
            render_search(search_index, index)


if __name__ == "__main__":
    main()
