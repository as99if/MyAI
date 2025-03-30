"""import google.generativeai as genai

def gemini_query(query):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(query)
    return response.text




def gemini_google_search(context_conversation_history: list = [],  max_retries: int = 3, message_prompt: str = ""):
    import json
    import logging
    import time
    import google.generativeai as genai
    from google.ai.generativelanguage_v1beta.types import content
    import google.ai.generativelanguage as glm

    with open("src/prompts/system_prompts.json", "r") as f:
        system_prompt = json.load(f)

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    genai.configure(api_key='YOUR_API_KEY')
    # Model configuration
    model = genai.GenerativeModel(
        model_name="gemini-1.0-pro",  # Using stable version
        generation_config={
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
            "stop_sequences": [],
            "candidate_count": 1
        },
        safety_settings={
            "harassment": "block_none",
            "hate_speech": "block_none",
            "sexually_explicit": "block_none",
            "dangerous_content": "block_none"
        },
        tools=[
            genai.protos.Tool(
                google_search=genai.protos.Tool.GoogleSearch(
                    enable_citations=True
                ),
            ),
        ],
    )

    def get_chat_response(session, message_prompt, retry_count=3):
        try:
            response = session.send_message(
                content=message_prompt,
                generation_config={
                    "temperature": 0.7,
                    "candidate_count": 1,
                    "stop_sequences": [],
                }
            )

            if not response.text:
                raise ValueError("Empty response received")

            return response.text

        except Exception as e:
            if retry_count < max_retries:
                logging.warning(
                    f"Retry {retry_count + 1}/{max_retries} after error: {str(e)}")
                time.sleep(1)  # Add delay between retries
                return get_chat_response(session, message_prompt, retry_count + 1)
            else:
                logging.error(f"Failed after {max_retries} retries: {str(e)}")
                raise

    try:
        # Initialize chat session
        chat_session = model.start_chat(history=context_conversation_history)

        # Get response with retry mechanism
        response_text = chat_session.get_chat_response(
            session=chat_session, message_prompt=message_prompt)
        # Validate JSON response
        try:
            response_json = json.loads(response_text)
            print(response_json)
            return response_json
        except json.JSONDecodeError:
            logging.error("Invalid JSON response")
            raise ValueError("Response not in valid JSON format")

    except Exception as e:
        logging.error(f"Error in Gemini API call: {str(e)}")
        raise

"""