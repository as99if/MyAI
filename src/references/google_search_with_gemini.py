"""
Install an additional SDK for JSON schema support Google AI Python SDK
$ pip install -q -U google-generativeai
$ pip install google.ai.generativelanguage
"""

import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
  tools = [
    genai.protos.Tool(
      google_search = genai.protos.Tool.GoogleSearch(),
    ),
  ],
)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "What is the weather today in Zurich?",
      ],
    },
  ]
)

response = chat_session.send_message("INSERT_INPUT_HERE")

print(response.text)