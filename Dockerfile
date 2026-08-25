FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Default command: show help
CMD ["python", "scripts/run_pipeline.py", "--help"]
