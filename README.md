# Autopodcast

An AI-powered podcast processing pipeline that automatically transcribes audio, segments content into chapters, generates summaries, and extracts keywords using advanced LLMs.

## Features

- **Audio Transcription**: Powered by OpenAI's Whisper model
- **Intelligent Segmentation**: Automatically divides content into meaningful chapters
- **AI Summarization**: Uses Google's Gemini 2.5 Flash for concise 4-5 sentence summaries
- **Keyword Extraction**: LLM-based topic and concept identification
- **Web Interface**: User-friendly Streamlit UI for easy interaction

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get it here](https://aistudio.google.com/app/apikey))
- Audio file in supported format (MP3, WAV, etc.)

## Installation



### 1. Install Required Packages

```bash
pip install -r requirements.txt
```


### 2. API Configuration

#### Create Environment File

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

#### Add Your Gemini API Key

Open `.env` and add your API key:

```env
GEMINI_API_KEY='API KEY HERE'
```

#### Get Your Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key
5. Paste it into your `.env` file


## Usage

### Running with Streamlit

Start the web interface:

```bash
streamlit ui/run app_streamlit.py
```

This will open your browser automatically at `http://localhost:8501`

### Using the Web Interface

1. **Upload Audio**: Click "Browse files" and select your podcast audio file
2. **Configure Settings** (optional):
   - Adjust Whisper model size (tiny, base, small, medium, large)
   - Set Gemini model (gemini-1.5-flash or gemini-1.5-pro)
3. **Process**: Click "Process Podcast" button
4. **View Results**: 
   - See transcription, chapters, summaries, and keywords
   - Download the generated PDF report



## Configuration

### Whisper Models

Available models (trade-off between speed and accuracy):
- `tiny` - Fastest, least accurate
- `base` - Fast, basic accuracy
- `small` - Balanced
- `medium` - Good accuracy (recommended)
- `large` - Best accuracy, slowest

### Gemini Models

- `gemini-2.0-flash` - Fast, cost-effective (recommended)
- `gemini-2.0-pro` - Higher quality, slower, more expensive



## How It Works

### 1. Transcription (Whisper)
```
Audio File → Whisper Model → Raw Transcript with Timestamps
```

### 2. Segmentation
```
Transcript → Topic Similarity Analysis → Chapters with Boundaries
```

### 3. Summarization (Gemini LLM)
```
Chapter Text → Smart Chunking (if >30k chars) → Gemini API → 4-5 Sentence Summary
```

**Chunking Strategy for Long Transcripts:**
- Splits text on sentence boundaries
- Max ~30,000 characters per chunk (~7,500 tokens)
- Summarizes each chunk independently
- Combines chunk summaries into final 4-5 sentences

### 4. Keyword Extraction (Gemini LLM)
```
Chapter Text → Gemini Analysis → 8 Topic-Based Keywords per Chapter
```

**LLM Advantage:**
- Identifies semantic concepts, not just word frequency
- Example: "machine learning", "neural networks" vs. "the", "and", "is"


## Troubleshooting

### API Key Issues

**Error**: `"Gemini API key not found"`

**Solution**: 
1. Ensure `.env` file exists in project root
2. Verify `GEMINI_API_KEY` is set in `.env`
3. Restart your application after adding the key

### Fallback to Simple Summarization

If you see: `"Using fallback summarization"`

**Causes**:
- Invalid API key
- API rate limits exceeded
- Network connectivity issues

**Solution**: Check your API key and internet connection

### Link to Github repo






