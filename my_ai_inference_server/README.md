## Inference Server


#### Install and run server
from the inference_server directory

```shell
pip install --upgrade pip
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
pip install -r requirements.txt
python -m serve
```

** note: running in docker not recommended for macos currently.

```shell
docker build -t my_ai_inference_server -f Dockerfile.inference_server .
# Or
docker build -t my_ai_inference_server -f Dockerfile.inference_server . --no-cache

# Run with environment variables from .env file
docker run -p 50001:50001 --env-file .env my_ai_inference_server

# Run in detached mode
docker run -d -p 50001:50001 --name mcp_server --env-file .env my_ai_inference_server

# If your container is named inference_server
docker exec inference_server env
# Or if you're using the container ID
docker exec <container_id> env
# If your container is named inference_server
docker inspect --format='{{range .Config.Env}}{{println .}}{{end}}' inference_server

# Or using the container ID
docker inspect --format='{{range .Config.Env}}{{println .}}{{end}}' <container_id>
# For example, to check GEMINI_API_KEY
docker exec inference_server sh -c 'echo $GEMINI_API_KEY'

# For example, to check GEMINI_API_KEY
docker exec inference_server sh -c 'echo $GEMINI_API_KEY'

```