FROM python:3.11-slim

# OpenCV and MediaPipe need these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/     app/
COPY routes/  routes/
COPY static/  static/

# Runtime directories (face.db and data/ are mounted via volumes in production)
RUN mkdir -p data

EXPOSE 8000

# Init DB schema then start the server
CMD ["sh", "-c", "python -m app.init_db && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
