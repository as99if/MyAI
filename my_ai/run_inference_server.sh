#!/bin/bash

# uvicorn --factory llama_cpp.server.app:create_app --host $HOST --port $PORT
python -m llama_cpp.server --host 0.0.0.0 --port 50001 --config_file "inference_server_config.json"
# python -m llama_cpp.server --host 0.0.0.0 --port 50001 --model "/Users/asifahmed/Development/MyAI/models/llm/Llama-3.2-3B-Instruct-Q8_0.gguf"


"python",
                "-m",
                "llama_cpp.server",
                "--model",
                f"{model_path}",
                "--chat_format",
                "gemma",
                "--port",
                "50001",
                "--use_mmap", "true",
                "--verbose", "true"