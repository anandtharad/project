# syntax=docker/dockerfile:1
# Self-hosted Laya decision server (laya-serve), CPU only. It speaks TypeSafe
# Jev's POST /v1/systemone protocol; LAYA_JEV_STRICT=1 (set in compose.yaml)
# returns the strict Jev response shape. Weights download from Hugging Face on
# first start into the laya-cache volume (~800 MB for the English checkpoint).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    USE_TF=0

# CPU-only torch first (the default Linux wheel bundles CUDA), then the
# pinned Laya release and its server extra.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1" \
    && pip install "laya[serve]==0.4.6"

RUN useradd --create-home --uid 10002 laya && mkdir -p /cache && chown laya /cache
USER laya

EXPOSE 8100
HEALTHCHECK --interval=15s --timeout=3s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=2).status == 200 else 1)"

CMD ["laya-serve"]
