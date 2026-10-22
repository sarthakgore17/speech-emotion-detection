FROM python:3.11-slim

WORKDIR /app

# Install system audio libraries (libsndfile & ffmpeg for decoding webm, wav, mp3, flac)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code and model
COPY . .

# Create uploads directory
RUN mkdir -p uploads && chmod 777 uploads

# Default port: 7860 for Hugging Face Spaces, fallback to $PORT on Render/Koyeb
ENV PORT=7860
EXPOSE 7860

# Run with Gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 2 --timeout 120 main:app"]
