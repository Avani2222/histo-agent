FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# CPU-only torch keeps the image small; inference on one patch is fast on CPU
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt
COPY src ./src
COPY configs ./configs
RUN pip install -e . --no-deps
# model + index are mounted or copied at deploy time (artifacts/, data/index/)
EXPOSE 8080
CMD ["sh", "-c", "uvicorn histo_agent.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
