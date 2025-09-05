FROM python:3.11-slim

# System deps for OpenCV + Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libtesseract-dev libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . /app

# Tesseract path + runtime port
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV PORT=8000

# Expose & start
EXPOSE 8000
CMD ["sh","-c","exec uvicorn extractor_api_lazyprod_ready:app --host 0.0.0.0 --port ${PORT:-8080} --workers 2 --access-log --proxy-headers --forwarded-allow-ips='*'"]
