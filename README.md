# Whisper Transcription API

This project deploys a fine-tuned Whisper transcription service as a FastAPI application. It accepts audio files, transcribes them using a fine-tuned Whisper model from Hugging Face, creates Word documents for each transcription, and returns a ZIP archive of the generated documents.

## Features

- Accepts multiple audio file formats (e.g., .mp3, .m4a, .wav)
- Converts audio to 16 kHz mono WAV for transcription
- Uses a fine-tuned Whisper model from Hugging Face
- Returns a ZIP file containing transcription Word documents

## Setup & Deployment

### Locally

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
