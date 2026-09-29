# Use official Python image with specific version
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository
COPY . .

# Set default command
CMD ["python", "scripts/run_pipeline.py", "--help"]
