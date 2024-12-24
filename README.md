    docker build -t Dockerfile.ollama .
    docker run -d --name ollama-server Dockerfile.ollama
    python3 -m venv .venv
    pip install --upgrade pip
    pip install -r requrements.txt


- speech recognition
- ollama server
- groq api
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
check SiLLM for training in mlx

check out https://github.com/developersdigest/ai-devices/tree/main

# pip install pyautogui gtts playsound SpeechRecognition requests openai applescript pillow dspy-ai
