# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set the working directory inside the container
WORKDIR /app/scripts

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and set permissions
RUN groupadd -r pipeline && useradd -r -g pipeline pipeline \
    && chown -R pipeline:pipeline /app

# Copy requirements file
COPY --chown=pipeline:pipeline requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=pipeline:pipeline . /app/

# Switch to non-root user
USER pipeline

# Default command (can be overridden by docker-compose)
CMD ["python", "main.py"]
