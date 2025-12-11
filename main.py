# main.py

import argparse
from autopodcast.pipeline import process_podcast

def main():
    parser = argparse.ArgumentParser(description="AutoPodcast – Podcast Indexing & Summarization")
    parser.add_argument("audio_path", help="Path to audio file (.mp3 / .wav)")
    args = parser.parse_args()

    index, search_index = process_podcast(args.audio_path)

    # For debugging: print basic info
    print(f"Processed: {index.audio.path}")
    print(f"Chapters: {len(index.chapters)}")
    for ch in index.chapters:
        print(f"[{ch.id}] {ch.start:.1f}s – {ch.end:.1f}s")
        print(f"Summary: {ch.summary}")
        print(f"Keywords: {', '.join(ch.keywords)}")
        print()

if __name__ == "__main__":
    main()
