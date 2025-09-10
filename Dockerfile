# Fake News Detection - Production Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs data evaluation_plots

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860 || exit 1

# Run application
CMD ["python", "web_application.py"]