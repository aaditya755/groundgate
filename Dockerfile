FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY groundgate/ ./groundgate/
COPY data/ ./data/
COPY eval/ ./eval/
COPY app/ ./app/
COPY LICENSE .

EXPOSE 8000

# Keys are NOT baked into the image: pass them at run time with --env-file .env
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"]