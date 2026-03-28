FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download default model
RUN python -c "from ultralytics import YOLO; YOLO('yolo26s.pt')"

EXPOSE 8000

# Default: run API server
CMD ["python", "cli.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
