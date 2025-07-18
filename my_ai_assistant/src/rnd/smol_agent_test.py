from typing import Optional

import requests
from smolagents import tool, OpenAIServerModel, ToolCallingAgent

from src.ai_tools.gemini import gemini_inference
from src.config.config import load_config

config = load_config()

# Configure connection to local llama.cpp server
model = OpenAIServerModel(
    model_id="gemma-3-1b",
    api_base="http://localhost:50001/v1",
    api_key="sk-anykey",  # Match your server's API key
    flatten_messages_as_text=True
)


# Create agent with Fibonacci tool
agent = ToolCallingAgent(
    model=model,
    tools=[
        gemini_inference
    ],
    max_steps=5
)

# Execute the agent

# agent.run("5000 dollars to Euros")
# agent.run("What is the weather in New York?")
# agent.run("Give me the top news headlines")
# agent.run("Tell me a joke")
# agent.run("Tell me a Random Fact")
# agent.run("who is Elon Musk?")

response = agent.run("What time is it in Japan right now? Ask Gemini.")
print(f"Agent response: {response}")