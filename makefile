# Makefile to build and run the servers

# Help target
help:
	@echo "Makefile to build and run the servers"
    

# Run the Docker container using docker-compose
db-run:
	@echo "Running the Docker container..."
	@docker-compose -f docker-compose.db.yml up
	

# Stop the Docker container
db-stop:
	@echo "Stopping the Docker container..."
	@docker-compose -f docker-compose.db.yml down