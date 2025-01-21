# Makefile to build and run the Ollama server


# Default target to build and run the Ollama server
ollama: build ollama-run

# Help target
help:
	@echo "Makefile to build and run the Ollama server"
	@echo ""
	@echo "Usage:"
	@echo "  make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  ollama      			 	- Build the Docker image and run the container"
	@echo "  ollama-build    			- Build the Docker image using the ollama-build-script.sh"
	@echo "  ollama-build-no-cache   	- Build the Docker (without using cache) image using the ollama-build-script.sh"
	@echo "  ollama-run      			- Run the Docker container using docker-compose"
	@echo "  ollama-stop     			- Stop the Docker container"
	@echo "  ollama-clean    			- Clean up the Docker environment (stop the container and remove the image)"
	@echo "  help     					- Display this help message"

# Build the Docker image using the ollama-build-script.sh
servers-build:
	@echo "Building the Docker image..."
	@docker-compose -f docker-compose.servers.yml build

servers-build-no-cahce:
	@echo "Building the Docker image..."
	@docker-compose -f docker-compose.servers.yml build --no-cache

# Run the Docker container using docker-compose
servers-run:
	@echo "Running the Docker container..."
	@docker-compose -f docker-compose.servers.yml up

# Stop the Docker container
servers-stop:
	@echo "Stopping the Docker container..."
	@docker-compose -f docker-compose.servers.yml down

# Clean up the Docker environment
servers-clean: stop
	@echo "Removing Docker images..."
	@docker rmi ollama_server_image

.PHONY: ollama ollama-build ollama-run ollama-stop ollama-clean