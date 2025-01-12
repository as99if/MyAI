put model name in .env
```shell
make ollama-build
make ollama-run
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
- check out https://deepinfra.com , https://deepinfra.com/meta-llama/Llama-3.2-11B-Vision-Instruct
- check out https://speechbrain.readthedocs.io/en/latest/tutorials/basics.html
- check out https://github.com/noshluk2/ROS2-Self-Driving-Car-AI-using-OpenCV
- check out https://github.com/t41372/Open-LLM-VTuber
- check out https://github.com/bigsk1/voice-chat-ai/tree/main
- check out https://github.com/chigkim/VOLlama
- * check out https://github.com/apeatling/ollama-voice-mac/tree/trunk
- * check out https://github.com/huggingface/speech-to-speech
- * check out https://github.com/ictnlp/LLaMA-Omni/tree/main
- check out https://github.com/jlonge4/local_llama/blob/main/local_llama_v3.py
https://github.com/mudler/LocalAI
speech engine https://github.com/netease-youdao/EmotiVoice

no stt, llm, and tts... any to any model connected with good llm and internet
- https://pytorch.org/hub/snakers4_silero-models_tts/
- https://github.com/janhq/WhisperSpeech/tree/main/ichigo-whisper
- https://huggingface.co/pyannote/segmentation
- https://huggingface.co/gpt-omni/mini-omni2
- https://github.com/suno-ai/bark
https://huggingface.co/facebook/seamless-m4t-v2-large
https://github.com/Leon-Sander/Local-Multimodal-AI-Chat

https://github.com/juliuskunze/speechless
https://github.com/uukuguy/speechless
check SiLLM for training in mlx

check out https://github.com/developersdigest/ai-devices/tree/main


Gemini Map explorer with maps and gemini api
** https://github.com/google-gemini/starter-applets/tree/main
- classification and function call for distances, waypoints etc.


*** https://github.com/microsoft/markitdown

# pip install pyautogui gtts playsound SpeechRecognition requests openai applescript pillow dspy-ai


nice one - https://github.com/browser-use/browser-use

sota stt tts


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
