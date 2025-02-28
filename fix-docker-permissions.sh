#!/bin/bash
# Script to fix Docker socket permissions inside Airflow containers

# Log start
echo "Starting Docker socket permission script..."

# Check if Docker socket exists
if [ -e /var/run/docker.sock ]; then
    echo "Found Docker socket at /var/run/docker.sock"
    
    # Get current permissions
    current_perms=$(stat -c "%a" /var/run/docker.sock 2>/dev/null || ls -la /var/run/docker.sock | awk '{print $1}')
    echo "Current permissions: $current_perms"
    
    # Change permissions to be world-readable and writable
    # This is done as a fallback if the group membership approach doesn't work
    chmod 666 /var/run/docker.sock
    
    # Verify the change
    new_perms=$(stat -c "%a" /var/run/docker.sock 2>/dev/null || ls -la /var/run/docker.sock | awk '{print $1}')
    echo "New permissions: $new_perms"
    
    echo "Docker socket permissions updated successfully."
else
    echo "ERROR: Docker socket not found at /var/run/docker.sock"
    exit 1
fi 