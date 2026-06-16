# CPU / development image. Runs the tracker, tests and the synthetic demo.
# For GPU TensorRT deployment see Dockerfile.jetson.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && \
    pip install numpy scipy pyyaml opencv-python-headless

COPY . .
RUN pip install -e .

# Default to the no-dependency synthetic demo so `docker run` proves it works.
ENTRYPOINT ["python", "scripts/demo_synthetic.py"]
CMD ["--targets", "60", "--frames", "300"]
