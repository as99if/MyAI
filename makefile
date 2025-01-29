# Makefile to build and run the servers

# Define log and PID file locations
INFERENCE_LOG := ./logs/inference-server.log
INFERENCE_PID := ./logs/inference-server.pid

# Help target
help:
	@echo "Makefile to build and run the Ollama server"



run-inference-server-daemon:
	@touch $(INFERENCE_PID)
	@if [ ! -f run_inference_server.sh ]; then \
        echo "Error: run_inference_server.sh not found"; \
        exit 1; \
    fi
	@echo "Starting inference server as daemon..."
	@chmod +x run_inference_server.sh
	@nohup ./run_inference_server.sh > $(INFERENCE_LOG) 2>&1 & echo $$! > $(INFERENCE_PID)
	@echo "Inference server started. PID: $$(cat $(INFERENCE_PID))"
	@echo "Logs available at: $(INFERENCE_LOG)"

status-inference-server-daemon:
	@if [ -f $(INFERENCE_PID) ]; then \
        PID=$$(cat $(INFERENCE_PID)); \
        if ps -p $$PID > /dev/null; then \
            echo "Inference server is running (PID: $$PID)"; \
            ps -p $$PID -o pid,ppid,%cpu,%mem,start,time,command; \
            echo "\nLast 5 log lines:"; \
            tail -n 5 $(INFERENCE_LOG); \
        else \
            echo "PID file exists but process is not running"; \
            rm -f $(INFERENCE_PID); \
        fi; \
    else \
        echo "Inference server is not running"; \
    fi

stop-inference-server-daemon:
	@if [ -f $(INFERENCE_PID) ]; then \
        echo "Stopping inference server..."; \
        kill $$(cat $(INFERENCE_PID)) || true; \
        rm -f $(INFERENCE_PID); \
        echo "Inference server stopped."; \
    else \
        echo "No inference server PID file found."; \
    fi

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

.PHONY: servers-build servers-run servers-stop servers-clean