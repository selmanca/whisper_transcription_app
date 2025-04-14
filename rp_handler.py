import os
import base64
import subprocess
import tempfile
import wave
import contextlib
import webrtcvad
import torch
import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from docx import Document
import runpod

# Use the system temporary directory
tmpdir = tempfile.gettempdir()

# ------------------------
# Audio Conversion Helpers
# ------------------------

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
        # Make sure the audio file is 16kHz mono 16-bit PCM
        assert wf.getframerate() == 16000, "Sample rate must be 16kHz"
        assert wf.getnchannels() == 1, "Audio must be mono"
        assert wf.getsampwidth() == 2, "Audio must be 16-bit"
        pcm_data = wf.readframes(wf.getnframes())
    return pcm_data, 16000

def write_wave(path, audio_bytes, sample_rate=16000):
    """Write PCM data to a WAV file with a given sample rate."""
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)

# ------------------------
# VAD-Based Chunking Logic
# ------------------------

def vad_chunks_to_flexible_chunks(wav_path, aggressiveness=2):
    """
    Split the WAV audio into chunks using voice activity detection.
    
    The function divides the audio into frames, labels each frame as speech or not,
    and then groups frames that contain speech into chunks of configurable duration.
    """
    FRAME_MS = 30     # Frame length in milliseconds (must be 10, 20, or 30 for webrtcvad)
    MIN_SEC = 25      # Minimum chunk duration in seconds
    MAX_SEC = 30      # Maximum chunk duration in seconds

    audio_bytes, sr = read_wave(wav_path)
    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(sr * (FRAME_MS / 1000.0) * 2)  # 2 bytes per sample
    frames = [audio_bytes[i:i+frame_size] for i in range(0, len(audio_bytes), frame_size)]

    # Determine if each frame contains speech
    speech_frames = []
    for frame in frames:
        if len(frame) < frame_size:
            continue  # Skip incomplete frames at the end
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
            # Even when no speech is detected, add some silence for continuity
            current.extend(b'\x00' * frame_size)
            sec += FRAME_MS / 1000
        
        # End the chunk if it has reached the maximum duration
        # or if enough speech has been accumulated and the next frame is silence (or it's the last frame)
        if sec >= MAX_SEC or (sec >= MIN_SEC and (i + 1 == len(speech_frames) or speech_frames[i + 1] is None)):
            end_time = start_time + sec
            # Expand the chunk slightly (1.2 seconds of padding) at both sides if possible
            start_sample = max(0, int(start_time * sr) * 2 - int(1.2 * sr) * 2)
            end_sample = min(len(audio_bytes), int(end_time * sr) * 2 + int(1.2 * sr) * 2)
            chunk = audio_bytes[start_sample:end_sample]
            chunks.append(chunk)
            chunk_infos.append((start_time, end_time))
            current = bytearray()  # reset buffer for next chunk
            start_time = end_time
            sec = 0
    if current:
        end_time = start_time + sec
        chunks.append(bytes(current))
        chunk_infos.append((start_time, end_time))

    # Save the chunks to temporary wav files
    paths = []
    for i, (chunk, (start, end)) in enumerate(zip(chunks, chunk_infos)):
        path = os.path.join(tmpdir, f"vad_chunk_{i}.wav")
        write_wave(path, chunk, sr)        
        paths.append(path)
    return paths

def merge_transcriptions(transcriptions, overlap_words=15):
    """
    Merge a list of transcriptions by checking overlap between adjacent chunks.
    
    If the last few words of one chunk match the first few words of the next,
    they are merged to avoid duplicate content.
    """
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

# ------------------------
# Model Loading & Preparation
# ------------------------

# Load model name from environment variable, and use a default if not provided.
model_name = os.getenv("WHISPER_MODEL_NAME")
# print(f"Using Whisper model: {model_name}")

# Load the Whisper processor and model.
processor = WhisperProcessor.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

# ------------------------
# RunPod Handler Function
# ------------------------

def handler(event):
    """
    RunPod serverless handler that:
      - accepts one or more audio files (base64 encoded or file paths),
      - converts each to a 16kHz mono WAV,
      - splits each into speech-based chunks using VAD,
      - transcribes each chunk with Whisper,
      - merges the chunk transcriptions per audio file, and
      - merges all audio transcriptions into a single DOCX file (with line separators between them).
    """
    job_input = event.get("input", {}) or {}
    
    # Support both a single file or a list of files
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
    
    # Process each audio file provided as base64.
    for idx, audio_str in enumerate(audio_b64):
        original_audio = os.path.join(tmpdir, f"input_audio_original_{idx}")
        processed_audio = os.path.join(tmpdir, f"input_audio_processed_{idx}.wav")
        try:
            # Remove data URL header if present
            if audio_str.startswith("data:"):
                audio_str = audio_str.split(",", 1)[1]
            with open(original_audio, "wb") as f:
                f.write(base64.b64decode(audio_str))
        except Exception as e:
            return {"error": f"Invalid base64 audio data for audio index {idx}: {e}"}
        
        try:
            convert_audio_to_wav(original_audio, processed_audio)
        except Exception as e:
            return {"error": f"Audio conversion failed for audio index {idx}: {str(e)}"}
        
        try:
            # Perform VAD-based chunking and transcription
            chunk_paths = vad_chunks_to_flexible_chunks(processed_audio)
            transcriptions = []
            for i, chunk_path in enumerate(chunk_paths):
                waveform, _ = librosa.load(chunk_path, sr=16000)
                input_features = processor(
                    waveform,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.to(device)
                predicted_ids = model.generate(input_features)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                transcriptions.append(transcription.strip())
            final_text = merge_transcriptions(transcriptions)
            all_transcriptions.append(final_text)
        except Exception as e:
            return {"error": f"Transcription failed for audio index {idx}: {str(e)}"}
    
    # Process each audio file provided as a direct file path.
    for idx, path in enumerate(audio_path):
        processed_audio = os.path.join(tmpdir, f"input_audio_processed_path_{idx}.wav")
        try:
            convert_audio_to_wav(path, processed_audio)
        except Exception as e:
            return {"error": f"Audio conversion failed for audio file {path}: {str(e)}"}
        
        try:
            # Perform VAD-based chunking and transcription
            chunk_paths = vad_chunks_to_flexible_chunks(processed_audio)
            transcriptions = []
            for i, chunk_path in enumerate(chunk_paths):
                waveform, _ = librosa.load(chunk_path, sr=16000)
                input_features = processor(
                    waveform,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.to(device)
                predicted_ids = model.generate(input_features)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                transcriptions.append(transcription.strip())
            final_text = merge_transcriptions(transcriptions)
            all_transcriptions.append(final_text)
        except Exception as e:
            return {"error": f"Transcription failed for audio file {path}: {str(e)}"}
    
    # Merge all audio transcriptions into one document using a separator (here using two newlines and a dashed line)
    separator = "\n\n"
    combined_transcription = separator.join(all_transcriptions)
    
    # Save the combined transcription to a DOCX file
    doc = Document()
    doc.add_paragraph(combined_transcription.strip())
    output_docx = os.path.join(tmpdir, "transcription.docx")
    doc.save(output_docx)
    
    # Encode DOCX file as base64 for the response
    try:
        with open(output_docx, "rb") as f:
            docx_b64 = base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        return {"error": f"Could not encode DOCX file: {e}"}
    
    return {"result": docx_b64}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
