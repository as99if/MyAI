## MyAI MCP Server

- install
```shell
pip install --upgrade pip
pip install -r requirements.txt

python -m 
```

```shell
docker build -t my_ai_mcp_server -f Dockerfile.mcp_server .
# Or
docker build -t my_ai_mcp_server -f Dockerfile.mcp_server . --no-cache

# Run with environment variables from .env file
docker run -p 50002:50002 --env-file .env my_ai_mcp_server

# Run in detached mode
docker run -d -p 50002:50002 --name mcp_server --env-file .env my_ai_mcp_server

# If your container is named mcp_server
docker exec mcp_server env
# Or if you're using the container ID
docker exec <container_id> env
# If your container is named mcp_server
docker inspect --format='{{range .Config.Env}}{{println .}}{{end}}' mcp_server

# Or using the container ID
docker inspect --format='{{range .Config.Env}}{{println .}}{{end}}' <container_id>
# For example, to check GEMINI_API_KEY
docker exec mcp_server sh -c 'echo $GEMINI_API_KEY'

# For example, to check GEMINI_API_KEY
docker exec mcp_server sh -c 'echo $GEMINI_API_KEY'

```