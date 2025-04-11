import os
import base64
import subprocess
import runpod
import torch
import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from docx import Document

# Load environment variable for model name
model_name = os.getenv("WHISPER_MODEL_NAME")
print(f"Using Whisper model: {model_name}")

# Load model and processor
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Helper function: convert any audio to 16kHz mono WAV
def convert_audio_to_wav(input_path, output_path):
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-f", "wav", output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode('utf-8')}")

# Chunking function with overlap
def chunk_audio(audio_path, chunk_length=30, overlap=3):
    waveform, sr = librosa.load(audio_path, sr=16000)
    chunk_samples = int(chunk_length * sr)
    overlap_samples = int(overlap * sr)
    total_samples = len(waveform)

    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[start:end]
        chunk_path = f"/tmp/chunk_{start}.wav"
        sf.write(chunk_path, chunk, sr)
        chunks.append(chunk_path)
        start += chunk_samples - overlap_samples

    return chunks

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

    # Decode base64
    if audio_b64:
        try:
            if audio_b64.startswith("data:"):
                audio_b64 = audio_b64.split(",", 1)[1]
            with open(original_audio, "wb") as f:
                f.write(base64.b64decode(audio_b64))
        except Exception as e:
            return {"error": f"Invalid base64 audio data: {e}"}
    elif audio_path:
        original_audio = audio_path

    try:
        convert_audio_to_wav(original_audio, processed_audio)
    except Exception as e:
        return {"error": f"Audio conversion failed: {str(e)}"}

    try:
        chunk_paths = chunk_audio(processed_audio)
        final_text = ""

        for chunk_path in chunk_paths:
            input_features = processor(
                librosa.load(chunk_path, sr=16000)[0],
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(model.device)

            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            final_text += transcription.strip() + "\n"

    except Exception as e:
        return {"error": f"Transcription failed: {str(e)}"}

    doc = Document()
    doc.add_paragraph(final_text.strip())
    output_path = "/tmp/transcription.docx"
    doc.save(output_path)

    with open(output_path, "rb") as f:
        docx_b64 = base64.b64encode(f.read()).decode('utf-8')

    return {"result": docx_b64}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
