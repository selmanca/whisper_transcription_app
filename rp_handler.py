import os, base64
import runpod
from transformers import pipeline
from docx import Document

# Load the Whisper model pipeline globally at startup (uses Hugging Face Transformers)
whisper_pipeline = pipeline(
    "automatic-speech-recognition",
    model="sarumanca/whisper-finetuned"
)
# The model will be downloaded from Hugging Face on the first run if not cached.

def handler(event):
    """
    RunPod handler function to process a transcription job.
    Expects event['input']['audio_base64'] or event['input']['audio'].
    """
    job_input = event.get("input", {}) or {}
    audio_b64 = job_input.get("audio_base64")
    audio_path = job_input.get("audio")  # URL or file path if provided
    audio_file = None

    # If audio is provided as base64 string, decode it to a temp file
    if audio_b64:
        try:
            # Remove any data URL prefix if present
            if audio_b64.startswith("data:"):
                audio_b64 = audio_b64.split(",", 1)[1]
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return {"error": f"Invalid base64 audio data: {e}"}
        # Write bytes to a temporary file
        os.makedirs("/tmp", exist_ok=True)
        audio_file = "/tmp/input_audio"
        with open(audio_file, "wb") as f:
            f.write(audio_bytes)
    elif audio_path:
        # If a direct path or URL is given (not typical for this frontend), use it
        audio_file = audio_path
    else:
        return {"error": "No audio input provided."}

    # Run transcription using the Whisper pipeline
    try:
        result = whisper_pipeline(audio_file)
        # The pipeline returns a dict or list with the transcribed text under 'text'
        text = result["text"] if isinstance(result, dict) else result[0].get("text", "")
    except Exception as e:
        # Return error (RunPod will mark the job as failed if an "error" key is present)
        return {"error": str(e)}

    # Create a Word document with the transcribed text
    doc = Document()
    doc.add_paragraph(text)
    output_path = "/tmp/transcription.docx"
    doc.save(output_path)

    # Read the Word file and encode in base64 to send back
    with open(output_path, "rb") as f:
        docx_bytes = f.read()
    docx_b64 = base64.b64encode(docx_bytes).decode('utf-8')

    # Return the base64 string as result; RunPod will include this in the output JSON
    return {"result": docx_b64}

# Start the RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
