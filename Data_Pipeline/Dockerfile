# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1


# Set the working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install pandera separately to ensure it's properly installed
RUN pip install --no-cache-dir 'pandera>=0.15.1,<0.16.0'

# Copy the rest of the application
COPY . /app/

# Install the package in development mode to support both import styles
WORKDIR /
COPY .. /workspace/
RUN pip install -e /workspace/

# Set working directory for scripts
WORKDIR /app/scripts

# Default command (can be overridden by docker-compose)
CMD ["python", "main.py"]
