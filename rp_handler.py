import os
import base64
import subprocess
import tempfile
import wave
import contextlib
import webrtcvad
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from docx import Document
import runpod
import re
from openai import OpenAI

# Use the system temporary directory
tmpdir = tempfile.gettempdir()

# ----- Audio Processing Functions -----

def convert_audio_to_wav(input_path, output_path):
    """Convert any audio file to a 16kHz mono WAV using ffmpeg."""
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-f", "wav", output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode('utf-8')}")

def read_wave(path):
    """Read a WAV file and return its PCM data and sample rate."""
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        # Ensure audio is 16kHz, mono, 16-bit PCM.
        assert wf.getframerate() == 16000, "Sample rate must be 16kHz"
        assert wf.getnchannels() == 1, "Audio must be mono"
        assert wf.getsampwidth() == 2, "Audio must be 16-bit"
        pcm_data = wf.readframes(wf.getnframes())
    return pcm_data, 16000

def write_wave(path, audio_bytes, sample_rate=16000):
    """Write PCM data to a WAV file with given sample rate."""
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)

def vad_chunks_to_flexible_chunks(wav_path, aggressiveness=2):
    """
    Split the WAV audio into chunks using voice activity detection.
    Chunks will be between MIN_SEC and MAX_SEC seconds in length.
    """
    FRAME_MS = 30     # must be 10, 20, or 30 for webrtcvad
    MIN_SEC = 25
    MAX_SEC = 30
    audio_bytes, sr = read_wave(wav_path)
    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(sr * (FRAME_MS / 1000.0) * 2)  # 2 bytes per sample
    frames = [audio_bytes[i:i+frame_size] for i in range(0, len(audio_bytes), frame_size)]
    
    # Detect speech frames
    speech_frames = []
    for frame in frames:
        if len(frame) < frame_size:
            continue
        if vad.is_speech(frame, sr):
            speech_frames.append(frame)
        else:
            speech_frames.append(None)
    
    # Group frames into chunks
    chunks = []
    current = bytearray()
    sec = 0
    start_time = 0
    chunk_infos = []
    for i, f in enumerate(speech_frames):
        if f is not None:
            current.extend(f)
            sec += FRAME_MS / 1000
        elif current:
            current.extend(b'\x00' * frame_size)
            sec += FRAME_MS / 1000
        
        if sec >= MAX_SEC or (sec >= MIN_SEC and (i+1 == len(speech_frames) or speech_frames[i+1] is None)):
            end_time = start_time + sec
            start_sample = max(0, int(start_time * sr) * 2 - int(1.2 * sr) * 2)
            end_sample = min(len(audio_bytes), int(end_time * sr) * 2 + int(1.2 * sr) * 2)
            chunk = audio_bytes[start_sample:end_sample]
            chunks.append(chunk)
            chunk_infos.append((start_time, end_time))
            current.clear()
            start_time = end_time
            sec = 0
    if current:
        end_time = start_time + sec
        chunks.append(bytes(current))
        chunk_infos.append((start_time, end_time))
    
    # Save chunks to temporary WAV files
    paths = []
    for i, (chunk, (start, end)) in enumerate(zip(chunks, chunk_infos)):
        path = os.path.join(tmpdir, f"vad_chunk_{i}.wav")
        write_wave(path, chunk, sr)
        print(f"Chunk {i}: {start:.2f}s to {end:.2f}s -> {path}")
        paths.append(path)
    return paths

def merge_transcriptions(transcriptions, overlap_words=15):
    """Merge transcription strings; remove overlaps at chunk boundaries."""
    if not transcriptions:
        return ""
    merged = transcriptions[0].strip()
    for current in transcriptions[1:]:
        current = current.strip()
        m_words = merged.split()
        c_words = current.split()
        max_ov = 0
        for size in range(overlap_words, 0, -1):
            if m_words[-size:] == c_words[:size]:
                max_ov = size
                break
        merged += " " + " ".join(c_words[max_ov:])
    return merged

def add_markdown_paragraph(doc, markdown_text):
    """
    Parses a Markdown string and adds it as a paragraph to the DOCX Document.
    Text wrapped in ** is formatted in bold.
    """
    paragraph = doc.add_paragraph()
    # Split text into parts based on bold markers.
    parts = re.split(r'(\*\*.*?\*\*)', markdown_text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        else:
            paragraph.add_run(part)

# ----- RunPod Handler Function -----

def handler(event):
    """
    RunPod serverless handler that:
      - Accepts one or more audio files (as base64 or file paths),
      - Converts each to 16kHz mono WAV and splits into chunks using VAD,
      - Transcribes each chunk via Whisper,
      - Merges the transcriptions,
      - Sends the merged text to OpenAI's GPT-4.1 model to fix errors,
      - Parses the Markdown output (making headings bold) and creates a DOCX file,
      - Returns the DOCX file as a base64‑encoded string.
    """
    job_input = event.get("input", {}) or {}
    
    # Support single or multiple files for both base64 and path
    audio_b64 = job_input.get("audio_base64")
    audio_path = job_input.get("audio")
    
    if isinstance(audio_b64, str):
        audio_b64 = [audio_b64]
    elif audio_b64 is None:
        audio_b64 = []
    
    if isinstance(audio_path, str):
        audio_path = [audio_path]
    elif audio_path is None:
        audio_path = []
    
    if not (audio_b64 or audio_path):
        return {"error": "No audio input provided."}
    
    all_transcriptions = []
    
    # Process base64 audio files.
    for idx, audio_str in enumerate(audio_b64):
        original_audio = os.path.join(tmpdir, f"input_audio_original_{idx}")
        processed_audio = os.path.join(tmpdir, f"input_audio_processed_{idx}.wav")
        try:
            if audio_str.startswith("data:"):
                audio_str = audio_str.split(",", 1)[1]
            with open(original_audio, "wb") as f:
                f.write(base64.b64decode(audio_str))
        except Exception as e:
            return {"error": f"Invalid base64 audio data for index {idx}: {e}"}
        try:
            convert_audio_to_wav(original_audio, processed_audio)
        except Exception as e:
            return {"error": f"Audio conversion failed for index {idx}: {e}"}
        try:
            chunk_paths = vad_chunks_to_flexible_chunks(processed_audio)
            transcriptions = []
            for i, chunk_path in enumerate(chunk_paths):
                waveform, _ = librosa.load(chunk_path, sr=16000)
                processor = WhisperProcessor.from_pretrained(os.getenv("WHISPER_MODEL_NAME"))
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = WhisperForConditionalGeneration.from_pretrained(os.getenv("WHISPER_MODEL_NAME")).to(device)
                input_feats = processor(
                    waveform,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.to(device)
                pred_ids = model.generate(input_feats)
                txt = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
                transcriptions.append(txt.strip())
            final_text = merge_transcriptions(transcriptions)
            all_transcriptions.append(final_text)
        except Exception as e:
            return {"error": f"Transcription failed for audio index {idx}: {e}"}
    
    # Process audio files provided as direct file paths.
    for idx, path in enumerate(audio_path):
        processed_audio = os.path.join(tmpdir, f"input_audio_processed_path_{idx}.wav")
        try:
            convert_audio_to_wav(path, processed_audio)
        except Exception as e:
            return {"error": f"Audio conversion failed for file {path}: {e}"}
        try:
            chunk_paths = vad_chunks_to_flexible_chunks(processed_audio)
            transcriptions = []
            for i, chunk_path in enumerate(chunk_paths):
                waveform, _ = librosa.load(chunk_path, sr=16000)
                processor = WhisperProcessor.from_pretrained(os.getenv("WHISPER_MODEL_NAME"))
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = WhisperForConditionalGeneration.from_pretrained(os.getenv("WHISPER_MODEL_NAME")).to(device)
                input_feats = processor(
                    waveform,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.to(device)
                pred_ids = model.generate(input_feats)
                txt = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
                transcriptions.append(txt.strip())
            final_text = merge_transcriptions(transcriptions)
            all_transcriptions.append(final_text)
        except Exception as e:
            return {"error": f"Transcription failed for file {path}: {e}"}
    
    # Merge all transcriptions from all files.
    separator = "\n\n"
    combined_transcription = separator.join(all_transcriptions)
    
    # ----- LLM Processing -----
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY environment variable must be set."}
    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are ChatGPT. You fix spelling, grammar, punctuation, and formatting errors in Turkish text in radiology report setting.\n"
        "Rules:\n"
        "Fix all errors but do not change the meaning.\n"
        "Use Markdown **bold** for headings or names (e.g., **Heading:**).\n"
        "Add a line break after every sentence.\n"
        "Don't add or remove any major content.\n"
        "Only output the corrected text — no explanations."
    )
    # Wrap the merged text in double quotes as required.
    user_message = f'"{combined_transcription}"'
    
    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.8
        )
        llm_output = response.output_text
    except Exception as e:
        return {"error": f"LLM processing failed: {e}"}
    
    # ----- Create DOCX File -----
    # Create a DOCX file containing the LLM-corrected text,
    # parsing the Markdown to ensure headings/names are bold.
    doc = Document()
    add_markdown_paragraph(doc, llm_output.strip())
    output_docx = os.path.join(tmpdir, "transcription.docx")
    doc.save(output_docx)
    
    # Encode the DOCX file as base64 for the response.
    try:
        with open(output_docx, "rb") as f:
            docx_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return {"error": f"Could not encode DOCX file: {e}"}
    
    return {"result": docx_b64}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
