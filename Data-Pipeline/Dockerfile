# Use an official lightweight Python image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app/scripts

# Copy only the necessary files (improves build speed)
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r "requirements.txt"

# Default command to run (can be overridden by docker-compose)
# CMD ["python", "main.py"]
