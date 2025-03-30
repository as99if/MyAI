#!/bin/bash

# Script to start Docker Compose services, capture logs, and append them to a file.

# Set the log file name
LOG_FILE="docker-compose.log"

# Ensure the log file exists or create it
touch "$LOG_FILE"

# Start the Docker Compose services in detached mode
docker-compose up -d

# Capture logs from each service and append them to the log file
echo "Capturing logs and appending to $LOG_FILE..."
for service in $(docker-compose ps -q); do
  echo "Capturing logs from container: $service"
  docker logs -f "$service" 2>&1 | tee -a "$LOG_FILE"
done

echo "All logs have been captured and appended to $LOG_FILE."
echo "You can view the logs using: tail -f $LOG_FILE"

exit 0