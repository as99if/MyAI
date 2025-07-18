
MyAI


video - https://www.gradio.app/docs
audio -. https://www.gradio.app/guides/conversational-chatbot


Lightweight local AI voice assistant with multiple functionalities, targetted for edge devices or macos.
Omni like software (eh?) - but with multiple LLM/VLM, ASR, TTS and Object detection models, with long term and short term memory etc.

Used stack:
- llama.cpp (llama-cpp-python) server
- whisper.cpp (pywhispercpp)
- kokoro-onnx
- redis

  
curl silero models for stt tts and vad
```shell
python3 -m venv .venv
brew install portaudio
brew install ffmpeg

# fedora
sudo dnf install portaudio portaudio-devel
sudo dnf install portaudio ffmpeg

pip install --upgrade pip
pip install -r requrements.txt
make db-run  # for redis dbs
python -m inference_server.serve # has both, text and multimodal model
# python -m main # in another terminal, or run inference server as daemon
python -m 
brew install mactop
sudo mactop
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


#### Run server in docker compose
```shell
chmod +x run_docker_compose_with_logs.sh
./run_docker_compose_with_logs.sh
```

docker-compose build
docker-compose up
docker-compose logs -f
docker-compose down
docker-compose ps


# python flet or smth. to create flutter app
# vision processing in client flutter app for iphone and macos
# rest api form vision-indeference with my_ai postprocess

https://pypi.org/project/instructor/


https://github.com/AK391/llama-3.2-3b-voice/tree/main
https://github.com/AK391/ollama-gradio
https://github.com/AK391/moshi


---
replace google (if good)
https://github.com/jina-ai/node-serp - not good

google cloud console api's (asif.drmc21 - project Gemini API)
(https://console.cloud.google.com/google/maps-apis/api-list?inv=1&invt=AboExg&project=gen-lang-client-0930623728)
(https://developers.google.com/maps/documentation/route-optimization/client-libraries#python)
https://github.com/googlemaps/google-maps-services-python

----------

... encrypt prompt meh

3. Tools - concurrent tool use and create markitdown and then response
3.0. connect/disconnect to internet
3.1. search and gather resources with gemini - tools **
3.2. search and gather resources with local and web browser - tools **
3.3. create response and information material (markdown, summary etc.) - ms markitdown *
3.4. searches iterate in background
3.5. object detection and distance mesurement with yolo-v9 and response as markdown.
3.6. expense and income record with voice command and ocr receipt management
- on finish save in conversation history, and notify tool caller
- process status shared with tool caller
---

- inference: speech recognition whisper.cpp
- text to speech kokoro
- concurrent and asynchronous processes
- conversation history, summarization process on application start (it's on schedule from the inside, backup will not change)
- base inference: llama.cpp (tool caller)
- tool inference: groq api
- tool inference: gemini api
- tool inference: deepseek
- tool: siri
- tool: rag (only local)
- function calling/api router (tool call and process response)
- vector storage as memory, the backup conversation history and knowledge bases
- base and tool - inference: vision models: ^ 
- vision processing: object detection
- vision processing: screen capture
- tool vision processing: video capture and process vision model response concurrently
- tool infernece: camera capture and process vision model response concurrently

-------------------------------


- osascript or linux scripts
- search engine api call and jina ai to scrape
- maps api call
- anthropic api
- kv caching

   - nos stable ollama or llama.cpp yet

- https://github.com/carloscdias/whisper-cpp-python/blob/main/whisper_cpp_python/whisper.py
- https://github.com/thewh1teagle/kokoro-onnx/blob/main/examples/play.py



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
