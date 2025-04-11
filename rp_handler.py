import os
import base64
import subprocess
import runpod
from docx import Document
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
import torch
import re

# Load environment variable for model
model_name = os.getenv("WHISPER_MODEL_NAME")
print(f"Using Whisper model: {model_name}")

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor and model
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# Load Whisper pipeline with better decoding parameters
whisper_pipeline = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    stride_length_s=5,
    generate_kwargs={
        "task": "transcribe",
        "temperature": 0.3,
        "num_beams": 5,
        "length_penalty": 1.0,
        "repetition_penalty": 2.0,
        "return_timestamps": False
    },
    device=0 if device == "cuda" else -1
)

# Optional cleanup function to reduce repeating phrases
def clean_repetitions(text):
    return re.sub(r'(\b\w+(?:\s+\w+){0,3})\s+(?:\1\s+){2,}', r'\1 ', text)

# Convert any audio file to 16kHz mono WAV
def convert_audio_to_wav(input_path, output_path):
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-f", "wav",
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode('utf-8')}")

# RunPod handler function
def handler(event):
    job_input = event.get("input", {}) or {}
    audio_b64 = job_input.get("audio_base64")
    audio_path = job_input.get("audio")

    if not (audio_b64 or audio_path):
        return {"error": "No audio input provided."}

    os.makedirs("/tmp", exist_ok=True)
    original_audio = "/tmp/input_audio_original"
    processed_audio = "/tmp/input_audio_processed.wav"

    if audio_b64:
        try:
            if audio_b64.startswith("data:"):
                audio_b64 = audio_b64.split(",", 1)[1]
            audio_bytes = base64.b64decode(audio_b64)
            with open(original_audio, "wb") as f:
                f.write(audio_bytes)
        except Exception as e:
            return {"error": f"Invalid base64 audio data: {e}"}
    elif audio_path:
        original_audio = audio_path

    try:
        convert_audio_to_wav(original_audio, processed_audio)
    except Exception as e:
        return {"error": f"Audio conversion failed: {str(e)}"}

    try:
        result = whisper_pipeline(processed_audio)
        raw_text = result["text"] if isinstance(result, dict) else result[0].get("text", "")
        text = clean_repetitions(raw_text)
    except Exception as e:
        return {"error": f"Transcription failed: {e}"}

    doc = Document()
    doc.add_paragraph(text)
    output_path = "/tmp/transcription.docx"
    doc.save(output_path)

    with open(output_path, "rb") as f:
        docx_b64 = base64.b64encode(f.read()).decode('utf-8')

    return {"result": docx_b64}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
