# Gemini API Setup Guide

## Getting Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Get API Key" or "Create API Key"
4. Copy your API key

## Setting Up the API Key

1. Open the `.env` file in the root directory of this project
2. Replace `your_gemini_api_key_here` with your actual API key:
   ```
   GEMINI_API_KEY=''
   ```
3. Save the file

## Optional: Choose a Different Model

By default, the project uses `gemini-1.5-flash` (fast and cost-effective).
If you want to use a more powerful model, uncomment and set in `.env`:
```
GEMINI_MODEL=gemini-1.5-pro
```

## Security Note

⚠️ **Never commit your `.env` file to git!** 
The `.gitignore` file is already configured to exclude it.

## Installing Dependencies

After setting up your API key, install the required packages:
```bash
pip install -r requirements.txt
```

## Testing the Setup

You can test if the API key is working by running your podcast processing pipeline. If there's an issue with the API key, the system will fall back to simple rule-based summarization and show a warning message.
