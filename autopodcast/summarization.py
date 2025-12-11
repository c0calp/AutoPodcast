# autopodcast/summarization.py

from __future__ import annotations
from typing import List
import re
import os

from .models import Chapter
from .config import CONFIG
from google import genai


# Approximate token limit for Gemini input (leaving room for prompt and response)
MAX_CHARS_PER_CHUNK = 30000 

def _simple_fallback_summary(text: str, max_sentences: int = 5) -> str:
    """Fallback summarization if Gemini API fails"""
    # naive sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
    return " ".join(sentences[:max_sentences])


def _chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """
    Split text into chunks that fit within token limits.
    Tries to split on sentence boundaries when possible.
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_chunk = ""
    
    for sentence in sentences:
        # If a single sentence is too long, split it by words
        if len(sentence) > max_chars:
            words = sentence.split()
            for word in words:
                if len(current_chunk) + len(word) + 1 <= max_chars:
                    current_chunk += (" " if current_chunk else "") + word
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = word
        # Normal sentence handling
        elif len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def summarize_text(text: str) -> str:
    """
    Summarize text using Google's Gemini model.
    Handles long transcripts by chunking them to fit within token limits.
    Falls back to simple summarization if API call fails.
    
    This function receives the transcribed text from each chapter
    and feeds it through the Gemini LLM for summarization.
    """
    if not text.strip():
        return ""
    
    try:
        # Get API key from config or environment
        api_key = CONFIG.summarization.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or configure it in CONFIG.summarization.gemini_api_key"
            )
        
        # Initialize Gemini client with API key
        client = genai.Client(api_key=api_key)
        
        # Chunk the text if it's too long
        chunks = _chunk_text(text)
        
        if len(chunks) == 1:
            # Text fits in one request
            prompt = f"""Please provide a concise summary of the following podcast transcript segment. 
Focus on the key points and main ideas discussed. Keep the summary to 4-5 sentences.

Transcript:
{text}

Summary:"""
        else:
            # For multiple chunks, summarize each and then combine
            print(f"Long transcript detected. Splitting into {len(chunks)} chunks for processing.")
            chunk_summaries = []
            
            for i, chunk in enumerate(chunks, 1):
                prompt = f"""Please provide a brief summary of the following podcast transcript segment (part {i} of {len(chunks)}). 
Focus on the key points discussed. Keep it concise (2-3 sentences).

Transcript:
{chunk}

Summary:"""
                
                response = client.models.generate_content(
                    model=CONFIG.summarization.gemini_model,
                    contents=prompt,
                )
                
                if response and response.text:
                    chunk_summaries.append(response.text.strip())
            
            # Now combine the chunk summaries into a final summary
            combined_summaries = " ".join(chunk_summaries)
            prompt = f"""Please provide a comprehensive summary combining these partial summaries of a podcast. 
Create a coherent 4-5 sentence summary that captures the main ideas.

Partial summaries:
{combined_summaries}

Final summary:"""
        
        # Call Gemini API to generate content
        response = client.models.generate_content(
            model=CONFIG.summarization.gemini_model,
            contents=prompt,
        )
        
        # Extract the summary from response
        if response and response.text:
            return response.text.strip()
        else:
            print("Warning: Empty response from Gemini, using fallback")
            return _simple_fallback_summary(text)
            
    except Exception as e:
        print(f"Error calling Gemini API: {e}. Using fallback summarization.")
        return _simple_fallback_summary(text)


def summarize_chapters(chapters: List[Chapter]) -> List[Chapter]:
    """Generate summaries for all chapters using Gemini"""
    for chapter in chapters:
        text = " ".join(s.text for s in chapter.segments)
        chapter.summary = summarize_text(text)
    return chapters
