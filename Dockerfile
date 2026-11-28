FROM python:3.13-slim-trixie

ENV PYTHONPATH=/opt/smart-search \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/var/cache/huggingface

WORKDIR /opt/smart-search
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.12.1+cpu' \
    && pip install --no-cache-dir -r requirements.txt 'transformers==5.9.0' \
    && pip check \
    && python -c "import torch; assert torch.version.cuda is None" \
    && mkdir -p /state "$HF_HOME" \
    && chown 10001:10001 /state "$HF_HOME"

RUN groupadd --gid 10001 smartsearch \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin smartsearch

COPY softline_search ./softline_search
WORKDIR /state
USER 10001:10001
EXPOSE 8765
CMD ["python", "-m", "softline_search", "demo", "--host", "0.0.0.0", "--port", "8765", "--device", "cpu", "--mode", "hybrid", "--recommendation-agent", "v3.2", "--catalog", "/snapshot/Mindbox.yml.gz.xml", "--index", "/snapshot/.search-index.sqlite3", "--public", "--warmup"]
