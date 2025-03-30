# Makefile to build and run the servers

# Help target
help:
	@echo "Makefile to build and run the servers"
    

# Build images for the Docker container using docker-compose
build:
	@echo "Running the Docker container..."
	@docker-compose -f docker-compose.yml build

# Run the Docker container
up:
	@echo "Running the Docker container..."
	@docker-compose -f docker-compose.yml up

# Stop the Docker container
stop:
	@echo "Stopping the Docker container..."
	@docker-compose -f docker-compose.yml down

# Run the tests
test:
	@echo "Running the tests..."
	@docker-compose -f docker-compose.yml run --rm test
	@docker-compose -f docker-compose.yml run --rm test ./run_tests.sh
	@docker-compose -f docker-compose.yml run --rm test ./run_tests.sh --coverage
	@docker-compose -f docker-compose.yml run --rm test ./run_tests.sh --coverage --html
	@docker-compose -f docker-compose.yml run --rm test ./run_tests.sh --coverage --html --xml
	@docker-compose -f docker-compose.yml run --rm test ./run_tests.sh --coverage --html --xml --json