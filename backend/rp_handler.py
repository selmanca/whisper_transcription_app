import os
import base64
import subprocess
import runpod
from transformers import pipeline
from docx import Document

# Load Whisper pipeline globally at startup
whisper_pipeline = pipeline(
    "automatic-speech-recognition",
    model="sarumanca/whisper-finetuned"
)

# Helper function: convert audio to Whisper-compatible format (16kHz, mono WAV)
def convert_audio_to_wav(input_path, output_path):
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn",                 # Disable video processing (if video data exists)
        "-acodec", "pcm_s16le",# Ensure PCM codec
        "-ar", "16000",        # Sample rate (16kHz)
        "-ac", "1",            # Mono audio
        "-f", "wav",           # Force WAV format
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # Capture FFmpeg error details for easier debugging
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

    # Decode base64 audio input
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

    # Convert audio to required format
    try:
        convert_audio_to_wav(original_audio, processed_audio)
    except Exception as e:
        return {"error": f"Audio conversion failed: {str(e)}"}
      
    # Transcribe audio
    try:
        result = whisper_pipeline(processed_audio)
        text = result["text"] if isinstance(result, dict) else result[0].get("text", "")
    except Exception as e:
        return {"error": f"Transcription failed: {e}"}

    # Create a Word document with transcription
    doc = Document()
    doc.add_paragraph(text)
    output_path = "/tmp/transcription.docx"
    doc.save(output_path)

    # Encode document in base64
    with open(output_path, "rb") as f:
        docx_b64 = base64.b64encode(f.read()).decode('utf-8')

    return {"result": docx_b64}

# Start RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
