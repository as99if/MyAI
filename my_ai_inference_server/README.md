## Inference Server

- llama-cpp-python
- gemma-3 1b and 4b multimodal

#### Install and run server
from the inference_server directory

```shell
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
pip install -r requirements.txt
python -m inference_server.serve
```