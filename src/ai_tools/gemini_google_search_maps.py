
"""_summary_

- custom_search_agent
"""
from asyncio import sleep
import html
import json
import pprint
from typing import Any
import urllib.parse
import os
import webbrowser
from bs4 import BeautifulSoup
import requests
import google.generativeai as genai

from src.ai_tools.groq import groq_inference
from src.ai_tools.ms_markitdown import generate_markdwon, get_markdown_content
from src.utils.utils import load_config

def embed_search_result(query:str):
    from urllib.parse import quote_plus
    # Encode the search query
    config = load_config()
    encoded_query = quote_plus(query)
    url = f"{config.get('google_custom_search_engine_url')}{encoded_query}"
    # "https://cse.google.com/cse?cx={search_engine_id}%23gsc.tab=0&gsc.q=Rust%20Programming"
    

    webbrowser.open(url)


    return 

def google_custom_search(query, api_key, custom_search_engine_id, num_results=10) -> dict:
    """
    Performs a search using the Google Custom Search JSON API.
    
    Args:
        query (str): The search query.
        api_key (str): Your Google Custom Search API key.
        custom_search_engine_id (str): The ID of your Custom Search Engine.
        num_results (int, optional): The number of results to return. Defaults to 10.

    Returns:
        list: A list of search result dictionaries, or None if there's an error.
    """
    from urllib.parse import urlencode
    
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": custom_search_engine_id,
        "num": num_results,
    }
    params = urlencode(params)

    try:
        response = requests.get(base_url, params)
        # response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        
        content = json.loads(response.content)
        with open("e.json", 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=4, ensure_ascii=False)
        items = content["items"]
        
        # print(response.keys())
        
        embed_search_result(query)
        return items
        

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        raise


def skeem_search_results(links, n: int = 10) -> list[dict]:
    """
    - Take first n links from search results
    - Get remaining links after n
    - Randomly select 3 from remaining if available
    - Process selected links to get content
    - Return combined results

    Args:
        search_results (_type_): _description_
        n (int, optional): _description_. Defaults to 10.

    Returns:
        list[dict]: _description_
    """
    import random
    skeemed_search_results = []
    
    # Take first n links
    first_n = links[:n]
    
    # Get remaining links
    remaining = links[n:]
    
    # Select random 3 from remaining if available
    random_links = random.sample(remaining, min(3, len(remaining)))
    
    # Process selected links
    
    try:
        for item in first_n:
            item.update({"content": get_markdown_content(item['link'])}) # test generate_markdwon(llm_client=groq or gemini, file_path=link, llm_model=deepseek or phi4)
            skeemed_search_results.append(item)
        for link in random_links:
            item.update({"content": get_markdown_content(item['link'])})
            skeemed_search_results.append(item)
            
    except Exception as e:
        print(f"Error processing link {link}: {e}")
    
    return skeemed_search_results
    
def custom_search_agent(query, google_custopm_search_api_key, google_custom_search_engine_id) -> dict:
    """
    Ask the internet
    # groq
    # google
    # llama or deepseek or mistral
    Args:
        query (str): 

    Returns:
        str:
    """
    response = None
    search_results = google_custom_search(query, google_custopm_search_api_key, google_custom_search_engine_id)
    # Get all links
    snippets = list(map(lambda item: {"title": item['title'], "link": item['link'], "snippet": item['snippet']}, search_results))
    
    skeemed_search_results = skeem_search_results(snippets, n=5)
    search_results_skeemed = { "seach_result_item_snippets": snippets, "skeemed_search_results":  skeemed_search_results}
    # add this to backup memory
    
    # response = create a search result response from the search_results_skeemed with groq
    prompt = f"Google search query: {str(query)},\nGoogle search result:\n{str(search_results_skeemed)}\n\nCreate a search result response from the given information. Put the most relevant information first. Put links as reference in text if necessary."
    system_prompt = "You are a helpful AI assistant. You are given a google search query and the search results. You need to create a search result response, which will be the summary of the given information. Extract the most useful information first. There could be bullte points, lists, tables, etc. Put the most relevant information first. Put links as reference in text if necessary."
    response, model = groq_inference(prompt, system_prompt)
    
    response = {"role": f"groq_assistant-{model}", "type": "search_result_summary", "content": response},
    
    return response

def google_maps_client(location):
    import googlemaps
    from datetime import datetime

    config = load_config()
    gmaps = googlemaps.Client(key=config.get('google_maps_api_key'))

    # Geocoding an address
    geocode_result = gmaps.geocode(location)

    # Look up an address with reverse geocoding
    reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))

    # Request directions via public transit
    now = datetime.now()
    directions_result = gmaps.directions("Sydney Town Hall",
                                        "Parramatta, NSW",
                                        mode="transit",
                                        departure_time=now)

    # Validate an address with address validation
    addressvalidation_result =  gmaps.addressvalidation(['1600 Amphitheatre Pk'], 
                                                        regionCode='US',
                                                        locality='Mountain View', 
                                                        enableUspsCass=True)

    # Get an Address Descriptor of a location in the reverse geocoding response
    address_descriptor_result = gmaps.reverse_geocode((40.714224, -73.961452))
    return geocode_result, reverse_geocode_result, directions_result, addressvalidation_result, address_descriptor_result



def embed_maps(location):
    config = load_config()
    encoded_location = urllib.parse.quote(location)
    iframe_src = f"https://www.google.com/maps/embed/v1/place?key={config.get('google_maps_api_key')}&q={encoded_location}"

    iframe_html = f"""
    
    <iframe
        id="embed-map"
        width="600"
        height="450"
        style="border:0"
        loading="lazy"
        allowfullscreen
        referrerpolicy="no-referrer-when-downgrade"
        src="{iframe_src}"
    >
    </iframe>
    
    """
    with open("src/ai_tools/maps.html", "w") as f:
        f.write(iframe_html)
        # wait
        webbrowser.open("src/ai_tools/maps.html")
    

    return 

def test_embed():
    location = "Eiffel Tower, Paris"
    embed_maps(location)


def close_embed():
    # check web browser or map iframe and close on voice command
    pass

def process_query(query):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(query)
    return response.text


def gemini_google_search(context_conversation_history: list = [],  max_retries: int = 3, message_prompt: str = ""):
    import os
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
