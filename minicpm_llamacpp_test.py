from llama_cpp import Llama
from PIL import Image
import llama_cpp.llama_chat_format as llama_chat_format
# Initialize model with multimodal capabilities
client = Llama(
    model_path="/Users/asifahmed/Development/ProjectKITT/llm/base_model/Model-7.6B-Q8_0.gguf",
)
print("Model loaded")
print(client)
# Load and prepare the image
image = Image.open('/Users/asifahmed/Desktop/Screenshot 2025-01-19 at 00.36.45.png').convert('RGB')

# Format messages for multimodal chat
messages = [
    {
        "role": "user",
        "content": f"{image}\nHey, describe this image.",
    }
]

# Create chat completion
response = client.create_chat_completion(
    messages=messages
)

print(response['choices'][0]['message']['content'])