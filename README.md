MyAI

Local AI voice assistant with functionalities, targetted for edge devices.

Used stack:
- llama-cpp-python
- whisper-cpp-python
- kokoro-onnx
- redis

```shell
make servers-build
make servers-run
```
  
curl silero models for stt tts and vad
```shell
python3 -m venv .venv
pip install --upgrade pip
pip install -r requrements.txt
make servers-run  # for ollama server and redis
python -m main
```

Run tests
```shell
python -m pytest tests/test_conversation_history_service.py -v

```
----


llama.cpp python
```shell
# Linux and Mac
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_METAL=on" \
  pip install llama-cpp-python
```




----------
some functionality ideas (research like):
1. encrypt prompt
2. connect to internet
3. search and gather resources
4. disconnect internet
5. create response and information material (markdown, smmary etc.)
6. iterate in background
6. save in knowledge base
6. respond when asked
---

- speech recognition
- ollama server
- https://github.com/ollama/ollama-python
- groq api
- gemini api
- maps
- google search
- function calling
- api router
- vision model
- https://dspy.ai to optimize prompt if configured
- conversation history, summarization of a part of the history on application start
- vector storage as memory conversation hitory and knowledge base
- object detection and distance mesurement with yolo-v9
- screen capture
- multiprocessing, thread pooling and asynchronous processes
- text to speech
-------------------------------


- osascript or linux scripts
- search engine api call and jina ai to scrape
- maps api call
- anthropic api
- kv caching

https://github.com/OpenBMB/MiniCPM-o/blob/main/web_demos/minicpm-o_2.6/model_server.py    - nos stable ollama or llama.cpp yet

*** https://github.com/OpenBMB/MiniCPM-o?tab=readme-ov-file#speech-conversation


https://github.com/OpenBMB/IoA

https://modelbest.feishu.cn/wiki/RnjjwnUT7idMSdklQcacd2ktnyN


- https://github.com/carloscdias/whisper-cpp-python/blob/main/whisper_cpp_python/whisper.py




Gemini Map explorer with maps and gemini api
** https://github.com/google-gemini/starter-applets/tree/main
- classification and function call for distances, waypoints etc.


*** https://github.com/microsoft/markitdown

# pip install pyautogui gtts playsound SpeechRecognition requests openai applescript pillow dspy-ai


nice one - https://github.com/browser-use/browser-use




macmos to see cpu gpu stats on macos

# Basic Docker Commands
# Build an image
docker build -t image-name:tag .

# Run a container
docker run -d --name container-name -p host-port:container-port image-name:tag

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a container
docker stop container-name

# Remove a container
docker rm container-name

# Remove an image
docker rmi image-name:tag

# View container logs
docker logs container-name

# Execute command in running container
docker exec -it container-name bash

# Pull an image from registry
docker pull image-name:tag

# Push image to registry
docker push image-name:tag

# Network commands
docker network create network-name
docker network ls
docker network rm network-name

# Volume commands
docker volume create volume-name
docker volume ls
docker volume rm volume-name

# Docker system commands
docker system prune -a  # Remove unused data
docker system df       # Show docker disk usage
