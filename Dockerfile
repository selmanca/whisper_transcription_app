# Start from an official Python slim image
FROM python:3.10-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \ 
    python3-dev \
    ffmpeg \
    libsndfile1-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY rp_handler.py .

# Start the RunPod handler
CMD ["python", "rp_handler.py"]
