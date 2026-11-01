
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY artifacts/lab7/ ./artifacts/lab7/
COPY data/processed/lab7_metadata.json ./data/processed/lab7_metadata.json

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
