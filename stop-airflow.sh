#!/bin/bash

# Set working directory to the script's directory
cd "$(dirname "$0")"

# Stop all running containers
echo "Stopping all Airflow services..."
docker-compose down

echo "All services stopped" 