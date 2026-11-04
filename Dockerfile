FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN python -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), sys.version" \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e ".[api]" \
    && python -m pip install --no-cache-dir \
        "transformers>=4.51.1,<5.0" \
        "peft>=0.15,<1.0" \
        "accelerate>=1.5,<2.0" \
        "safetensors>=0.5,<1.0"

EXPOSE 8000

CMD ["uvicorn", "finetunex.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
