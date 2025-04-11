# Dockerfile
# Start from an official Python slim image
FROM python:3.10-slim

# Install system dependencies: ffmpeg for audio decoding, libsndfile1 for pysoundfile, and git if you ever pull models directly
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your RunPod handler code
COPY rp_handler.py .

# Command to start the RunPod handler
CMD ["python", "rp_handler.py"]
