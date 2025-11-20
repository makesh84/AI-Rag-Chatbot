FROM python:3.11-slim

WORKDIR /app

# System deps for PDF + unstructured
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default port
ENV PORT=8000

# Do NOT run ingest here by default (can be heavy).
# You can uncomment this if your docs are bundled into image:
# RUN python -m app.ingest || true

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
