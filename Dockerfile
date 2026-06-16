# Portable CPU image for the ProGress UI.
#
# Deploy targets:
#   • Hugging Face Spaces (Docker SDK) — set the Space SDK to "docker".
#   • Any container host (Render / Fly / a VM / local) —
#     `docker run --gpus all -p 7860:7860 …` (omit --gpus all for CPU-only hosts).
#
# requirements.txt ships CUDA 11.8 torch; the wheel falls back to CPU when no
# GPU is present, so this image runs on both GPU and CPU hosts.
#
# For a gradio-SDK Space deployed with `gradio deploy`, this file is ignored;
# it only matters for the Docker path.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/tmp/hf \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /app

# build-essential covers any source-only transitive dependency; libsndfile/
# libgl are commonly needed by audio/score tooling. Kept minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["python", "app.py"]
