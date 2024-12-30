#!/bin/sh

# Function to install jq if not installed
install_jq() {
  if ! command -v jq > /dev/null 2>&1; then
    echo "jq is not installed. Installing jq..."
    apt-get update && apt-get install -y jq
  else
    echo "jq is already installed."
  fi
}



# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check if MODEL_NAME is set from .env
if [ -z "$MODEL_NAME" ]; then
    # Install jq if not installed
    install_jq
    # Read the model name from config.json if not set in .env
    MODEL_NAME=$(jq -r '.model_name' src/config.json)
fi

# Ensure MODEL_NAME is set
if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME is not set in either .env or src/config.json"
    echo "Model: $MODEL_NAME"
    exit 1
fi

# Start the Ollama app in the background
ollama serve &

# Wait for the Ollama app to be ready by checking the /api/version endpoint
until curl -s http://localhost:11434/api/version; do
  echo "Waiting for Ollama app to be ready..."
  sleep 2
done

# Check if the model is available using the API
if curl -s http://localhost:11434/api/tags | jq -e ".models[] | select(.name == \"$MODEL_NAME\")" > /dev/null; then
    echo "Model $MODEL_NAME is already available."
else
    echo "Model $MODEL_NAME is not available. Pulling the model..."
    ollama pull $MODEL_NAME
fi

# Disconnect internet from the Docker container
# echo "Disconnecting internet to create a sandbox environment..."
# ip route del default
# remove internet after running everything

echo "Ollama is now running in sandbox mode with model $MODEL_NAME"

# Keep the script running to keep the container alive
tail -f /dev/null
