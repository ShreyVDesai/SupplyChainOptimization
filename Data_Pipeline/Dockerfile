# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

# Set the working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY Data_Pipeline/requirements.txt ./requirements.txt
COPY requirements.txt /app/

# Install Python packages
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application
COPY . /app/

# Set working directory for scripts
WORKDIR /app/scripts

# Default command (can be overridden by docker-compose)
CMD ["python", "main.py"]
