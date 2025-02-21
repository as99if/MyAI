# Makefile to build and run the servers

# Define log and PID file locations
LOG_FILE := ./logs/inference-server.log
PID_FILE := ./logs/inference-server.pid
SERVER_SCRIPT := SERVER_SCRIPT="python -m llama_cpp.server --host 0.0.0.0 --port 50001 --config_file \"inference_server_config.json\""
SERVER_PORT := 50001
# Help target
help:
	@echo "Makefile to build and run the Ollama server"

run:
	@echo "Starting llama.cpp server..."
	$(SERVER_SCRIPT) > $(LOG_FILE) 2>&1 & echo $$! > $(PID_FILE)
	@echo "Server started. See logs in $(LOG_FILE) and process ID in $(PID_FILE)"

status:
	@if [ -f "$(PID_FILE)" ]; then \
		pid=$$(cat $(PID_FILE)); \
		if kill -0 "$$pid" > /dev/null 2>&1; then \
			echo "Server is running (PID: $$pid). Checking port..."; \
			if nc -z localhost $(SERVER_PORT); then \
			  echo "Port $(SERVER_PORT) is also open."; \
      else \
			  echo "Port $(SERVER_PORT) is not accessible. Likely an issue with the server"; \
      fi; \
		else \
			echo "Server is NOT running (PID file exists but process is not running)"; \
		fi; \
	else \
		echo "Server is NOT running (no PID file found)"; \
	fi

logs:
	@echo "Showing server logs..."
	@tail -f $(LOG_FILE)

stop:
	@if [ -f "$(PID_FILE)" ]; then \
		pid=$$(cat $(PID_FILE)); \
		echo "Stopping server (PID: $$pid)..."; \
		kill "$$pid" 2>/dev/null; \
		sleep 1; \
		kill -0 "$$pid" 2>/dev/null && (kill -9 "$$pid" && echo "Server killed forcefully");\
    rm $(PID_FILE); \
		echo "Server stopped."; \
	else \
		echo "No server to stop (no PID file found)"; \
	fi

clean:
	@rm -f $(LOG_FILE) $(PID_FILE)
	@echo "Cleaned log and pid file"
    

# Run the Docker container using docker-compose
redis-run:
	@echo "Running the Docker container..."
	@docker-compose -f docker-compose.servers.yml up
	

# Stop the Docker container
redis-stop:
	@echo "Stopping the Docker container..."
	@docker-compose -f docker-compose.servers.yml down

# Clean up the Docker environment
servers-clean: stop
	@echo "Removing Docker images..."
	@docker rmi 



# Targets





.PHONY: servers-build servers-run servers-stop servers-clean