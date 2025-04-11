# Whisper Audio Transcription API (RunPod + Hugging Face)

This project is a serverless audio transcription API built for [RunPod](https://www.runpod.io), using a fine-tuned model from Hugging Face. It accepts audio input (as base64 or file path), transcribes it to text, and returns the transcription as a base64-encoded `.docx` Word document.

---

## 🔧 Features

- Accepts audio in base64 or local file path
- Supports `.mp3`, `.wav`, and `.m4a` files (with conversion)
- Converts audio to Whisper-compatible format (16kHz, mono WAV)
- Transcribes using a fine-tuned Whisper model
- Returns transcription as a Word `.docx` document (base64 encoded)

