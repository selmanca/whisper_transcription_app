# Start from an official Python slim image
FROM python:3.10-slim

# Install system dependencies: ffmpeg for audio decoding, git (if needed for transformers), and others
RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY rp_handler.py . 

# (If there were additional src files or model files, they would be copied here)

# Command to start the RunPod handler (this will run the serverless loop)
CMD ["python", "rp_handler.py"]
