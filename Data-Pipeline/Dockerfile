# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and set permissions
RUN groupadd -r pipeline && useradd -r -g pipeline pipeline \
    && mkdir -p /app \
    && chown -R pipeline:pipeline /app

# Set the working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application
COPY . /app/

# Ensure correct ownership of copied files
RUN chown -R pipeline:pipeline /app

# Switch to non-root user
USER pipeline

# Set working directory for scripts
WORKDIR /app/scripts

# Default command (can be overridden by docker-compose)
CMD ["python", "main.py"]
