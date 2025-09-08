FROM python:3.12-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/ src/
COPY app/ app/
COPY web/ web/
COPY config.ini .
COPY requirements-serve.txt .

RUN python -m pip install --upgrade pip wheel setuptools \
 && pip install --no-cache-dir -r requirements-serve.txt

EXPOSE 4000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:4000/ || exit 1

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:4000", "app.main:app"]

