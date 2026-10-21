FROM node:22-bookworm-slim AS client-build

WORKDIR /app/client
COPY client/package.json client/package-lock.json ./
RUN npm ci
COPY client/ ./
RUN npm run build

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install --with-deps chromium

COPY server/ ./server/
COPY data/ ./data/
COPY --from=client-build /app/client/dist ./client/dist

ENV JOBS_FILE=data/jobs.json \
    CLIENT_DIST=client/dist

EXPOSE 8000
CMD ["sh", "-c", "uvicorn server.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
