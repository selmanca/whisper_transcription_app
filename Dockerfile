FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg git

# Install Python dependencies
RUN pip install --no-cache-dir torch torchaudio transformers fastapi uvicorn pydub python-docx accelerator

# Copy the app code into the container
COPY . .

# Expose the port (must match the port in uvicorn command)
EXPOSE 3000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
