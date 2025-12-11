# autopodcast/search.py

from typing import List
import numpy as np
import faiss  # or fallback to simple numpy search
from .models import Transcript, TranscriptSegment, SearchResult
from .embeddings import EmbeddingModel
from .config import CONFIG

class SemanticSearchIndex:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.index = None
        self.segment_meta: List[TranscriptSegment] = []

    def build(self, transcript: Transcript):
        """
        Build a vector index over short transcript chunks.
        """
        texts = [s.text for s in transcript.segments]
        embeddings = self.embedder.embed_texts(texts).astype("float32")

        # FAISS index
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.segment_meta = transcript.segments

    def search(self, query: str) -> List[SearchResult]:
        """
        Query the index and return top-k results with timestamps.
        """
        cfg = CONFIG.search
        query_emb = self.embedder.embed_texts([query]).astype("float32")
        faiss.normalize_L2(query_emb)
        scores, idxs = self.index.search(query_emb, cfg.top_k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], idxs[0]):
            seg = self.segment_meta[int(idx)]
            results.append(
                SearchResult(
                    chapter_id=-1,  # optionally filled later when you map segments → chapters
                    segment_start=seg.start,
                    segment_end=seg.end,
                    snippet=seg.text,
                    score=float(score)
                )
            )
        return results
