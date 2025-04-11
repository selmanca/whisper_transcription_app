import os
import base64
import subprocess
import runpod
import torch
import wave
import webrtcvad
import contextlib
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from docx import Document

# Load environment variable for model name
model_name = os.getenv("WHISPER_MODEL_NAME")
print(f"Using Whisper model: {model_name}")

# Load model and processor
processor = WhisperProcessor.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

def convert_audio_to_wav(input_path, output_path):
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-f", "wav", output_path
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def read_wave(path):
    """Reads a WAV file and returns (PCM_bytes, sample_rate)."""
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        sr = wf.getframerate()
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        pcm = wf.readframes(wf.getnframes())
    return pcm, sr

def write_wave(path, pcm_bytes, sample_rate):
    """Writes PCM bytes to a WAV file."""
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

def vad_segment_generator(wav_path, aggressiveness=3, frame_ms=30, padding_ms=300):
    """
    Yields (start_byte, end_byte) tuples for speech segments in the WAV file.
    Uses WebRTC VAD with a padding window to smooth boundaries.
    """
    pcm, sr = read_wave(wav_path)
    vad = webrtcvad.Vad(aggressiveness)
    bytes_per_frame = int(sr * (frame_ms / 1000.0) * 2)  # 2 bytes/sample

    # Split into frames
    frames = [pcm[i:i+bytes_per_frame] for i in range(0, len(pcm), bytes_per_frame)]
    is_speech = [vad.is_speech(f, sr) for f in frames]

    # Build speech segments with padding
    pad_frames = int(padding_ms / frame_ms)
    segments = []
    start = None
    for i, speech in enumerate(is_speech):
        if speech and start is None:
            start = max(0, i - pad_frames)
        elif not speech and start is not None:
            end = min(len(frames), i + pad_frames)
            segments.append((start * bytes_per_frame, end * bytes_per_frame))
            start = None
    # Handle if file ends while still in speech
    if start is not None:
        segments.append((start * bytes_per_frame, len(pcm)))

    return segments, sr

def handler(event):
    job_input = event.get("input", {}) or {}
    audio_b64 = job_input.get("audio_base64")
    audio_path = job_input.get("audio")
    if not (audio_b64 or audio_path):
        return {"error": "No audio input provided."}

    os.makedirs("/tmp", exist_ok=True)
    orig = "/tmp/input_orig"
    wav16 = "/tmp/input_16k.wav"

    # Decode or copy
    if audio_b64:
        data = audio_b64.split(",",1)[1] if audio_b64.startswith("data:") else audio_b64
        with open(orig, "wb") as f:
            f.write(base64.b64decode(data))
    else:
        orig = audio_path

    try:
        convert_audio_to_wav(orig, wav16)
    except Exception as e:
        return {"error": f"Conversion failed: {e}"}

    # VAD-based segmentation
    segments, sr = vad_segment_generator(wav16, aggressiveness=3, frame_ms=30, padding_ms=300)
    if not segments:
        # fallback to single segment
        segments = [(0, None)]

    transcriptions = []
    for idx, (start_b, end_b) in enumerate(segments):
        pcm, _ = read_wave(wav16)
        seg_pcm = pcm[start_b:end_b] if end_b else pcm[start_b:]
        seg_path = f"/tmp/segment_{idx}.wav"
        write_wave(seg_path, seg_pcm, sr)

        # load and transcribe
        waveform, _ = sf.read(seg_path, dtype='int16')
        input_feats = processor(waveform, sampling_rate=sr, return_tensors="pt").input_features.to(device)
        pred_ids = model.generate(input_feats)
        txt = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        transcriptions.append(txt)

    # simple join with newline; you can apply merge logic if needed
    final_text = "\n".join(transcriptions)

    # save to docx
    doc = Document()
    doc.add_paragraph(final_text)
    out = "/tmp/transcription.docx"
    doc.save(out)
    with open(out,"rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    return {"result": b64}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
