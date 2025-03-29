FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download models using raw GitHub URLs
RUN mkdir -p /app/models && \
    wget -O /app/models/best.pt "https://github.com/AmrrGamal/Models_apis/raw/main/Models_apis/models/best.pt" && \
    wget -O /app/models/best_count.pt "https://github.com/AmrrGamal/Models_apis/raw/main/Models_apis/models/best_count.pt"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]