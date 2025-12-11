# autopodcast/keywords.py

from __future__ import annotations
from typing import List
from collections import Counter
import re
import os
from .models import Chapter
from .config import CONFIG
from google import genai


STOPWORDS = {
    "the", "and", "to", "of", "in", "a", "for", "is", "it", "on", "that",
    "this", "with", "as", "at", "by", "from", "or", "an", "be", "are",
}


def _tokenize(text: str) -> List[str]:
    """Simple tokenization - used as fallback"""
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def _extract_keywords_simple(text: str, max_keywords: int = 8) -> List[str]:
    """Fallback keyword extraction using word frequency"""
    tokens = _tokenize(text)
    if not tokens:
        return []
    counts = Counter(tokens)
    return [w for w, _ in counts.most_common(max_keywords)]


def _chunk_text_for_keywords(text: str, max_chars: int = 30000) -> List[str]:
    """
    Split text into chunks for keyword extraction if needed.
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = ""
    
    for sentence in sentences:
        if len(sentence) > max_chars:
            words = sentence.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 <= max_chars:
                    current_chunk += (" " if current_chunk else "") + word
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = word
        elif len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def extract_keywords_with_llm(text: str, max_keywords: int = 8) -> List[str]:
    """
    Extract keywords from text using Google's Gemini model.
    Falls back to simple keyword extraction if API call fails.
    """
    if not text.strip():
        return []
    
    try:
        # Get API key from config or environment
        api_key = CONFIG.summarization.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable"
            )
        
        # Initialize Gemini client with API key
        client = genai.Client(api_key=api_key)
        
        # Chunk the text if it's too long
        chunks = _chunk_text_for_keywords(text)
        
        if len(chunks) == 1:
            # Text fits in one request
            prompt = f"""Analyze the following podcast transcript and extract the {max_keywords} most important keywords or key phrases.
These should be the main topics, concepts, or themes discussed.
Return ONLY the keywords as a comma-separated list, nothing else.

Transcript:
{text}

Keywords:"""
        else:
            # For multiple chunks, extract keywords from each and combine
            print(f"Long transcript for keywords. Processing {len(chunks)} chunks.")
            all_keywords = []
            
            for i, chunk in enumerate(chunks, 1):
                prompt = f"""Analyze this podcast transcript segment and extract the 5-6 most important keywords or key phrases.
Return ONLY the keywords as a comma-separated list.

Transcript:
{chunk}

Keywords:"""
                
                response = client.models.generate_content(
                    model=CONFIG.summarization.gemini_model,
                    contents=prompt,
                )
                
                if response and response.text:
                    # Parse keywords from response
                    chunk_keywords = [kw.strip() for kw in response.text.strip().split(',')]
                    all_keywords.extend(chunk_keywords)
            
            # Now get final keywords from all collected keywords
            combined = ", ".join(all_keywords)
            prompt = f"""From this list of keywords extracted from different parts of a podcast, select the {max_keywords} most important and representative keywords.
Return ONLY the keywords as a comma-separated list.

All keywords:
{combined}

Final {max_keywords} keywords:"""
        
        # Call Gemini API to generate content
        response = client.models.generate_content(
            model=CONFIG.summarization.gemini_model,
            contents=prompt,
        )
        
        # Extract keywords from response
        if response and response.text:
            # Parse the comma-separated keywords
            keywords = [kw.strip().lower() for kw in response.text.strip().split(',')]
            # Clean up any empty strings or extra formatting
            keywords = [kw for kw in keywords if kw and len(kw) > 1]
            return keywords[:max_keywords]
        else:
            print("Warning: Empty response from Gemini for keywords, using fallback")
            return _extract_keywords_simple(text, max_keywords)
            
    except Exception as e:
        print(f"Error calling Gemini API for keywords: {e}. Using fallback.")
        return _extract_keywords_simple(text, max_keywords)


def assign_keywords_to_chapters(chapters: List[Chapter]) -> List[Chapter]:
    """Extract keywords for each chapter using Gemini LLM"""
    for chapter in chapters:
        text = " ".join(s.text for s in chapter.segments)
        chapter.keywords = extract_keywords_with_llm(text, max_keywords=8)
    return chapters


def build_global_keywords(chapters: List[Chapter], top_n: int = 20) -> List[str]:
    """Build global keywords from all chapter keywords"""
    counter = Counter()
    for chapter in chapters:
        counter.update(chapter.keywords)
    return [kw for kw, _ in counter.most_common(top_n)]
