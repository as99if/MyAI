from openai import OpenAI


api_key = ""
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
  model="gpt-4o",
  messages=[],
  response_format={
    "type": "text"
  },
  temperature=1,
  max_completion_tokens=2048,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)