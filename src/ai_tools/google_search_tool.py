from asyncio import sleep
from datetime import datetime
import json
import pprint
import time
from typing import Any
import webbrowser
import requests
import random
from src.ai_tools.groq import groq_inference
from src.ai_tools.ms_markitdown import generate_markdown_and_summarize, get_markdown_content
from src.utils.utils import load_config, split_list
from urllib.parse import urlencode
import tracemalloc


def embed_search_result(query: str):
    from urllib.parse import quote_plus
    # Encode the search query
    config = load_config()
    encoded_query = quote_plus(query)
    url = f"{config.get('google_custom_search_engine_url')}{encoded_query}"
    # "https://cse.google.com/cse?cx={search_engine_id}%23gsc.tab=0&gsc.q=Rust%20Programming"

    webbrowser.open(url)

    return


def google_custom_search(query, google_custom_search_api_key, google_custom_search_engine_id, num_results=10) -> dict:
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

    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": google_custom_search_api_key,
        "cx": google_custom_search_engine_id,
        "num": num_results,
    }
    params = urlencode(params)

    try:
        with requests.get(base_url, params=params) as response:
            # Raise an HTTPError for bad responses (4xx or 5xx)
            response.raise_for_status()

            # Parse the JSON content
            content = response.json()
            
            with open("e.json", 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=4, ensure_ascii=False)
            items = content["items"]

            # embed search result in web browser
            # embed_search_result(query)
            return items

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        raise

# test


"""def test_google_custom_search():
    config = load_config()
    query = "what is the current temparature of Stuttgart?"
    api_key = config.get("google_custom_search_api_key")
    custom_search_engine_id = config.get("google_custom_search_engine_id")
    items = google_custom_search(query=query, google_custom_search_api_key=api_key, google_custom_search_engine_id=custom_search_engine_id)
    pprint.pprint(items, indent=2)

test_google_custom_search()"""


def skeem_search_results(query, snippets, n: int = 10) -> list[dict]:
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
    
    skeemed_search_results = []

    # Take first n links
    first_n = snippets[:n]

    # Get remaining links
    remaining = snippets[n:]

    # Select random 3 from remaining if available
    random_links = random.sample(remaining, min(3, len(remaining)))

    # Process selected links

    visited_links = []
    blocked_links = []
    for item in first_n:
        try:
            # test generate_markdwon(llm_client=groq or gemini, file_path=link, llm_model=deepseek or phi4)
            #content = get_markdown_content(item['link'])
            content = generate_markdown_and_summarize(
                main_prompt=query,
                file_path=item['link']
            )
            formatted_content = {
                "title": item['title'],
                "link": item['link'],
                "snippet": item['snippet'],
                "content": content
            }
            del content
            skeemed_search_results.append(formatted_content)
            visited_links.append({"title": item["title"], "link": item['link']})
            del formatted_content
        except Exception as e:
            formatted_content = {
                "title": item['title'],
                "link": item['link'],
                "snippet": item['snippet'],
                "content": "cannot process data from link"
            }
            skeemed_search_results.append(formatted_content)
            blocked_links.append({"title": item['title'], "link": item['link']})
            del formatted_content
            continue
    for link in random_links:
        try:
            #content = get_markdown_content(item['link'])
            content = generate_markdown_and_summarize(
                main_prompt=query,
                file_path=item['link']
            )
            formatted_content = {
                "title": item['title'],
                "link": item['link'],
                "snippet": item['snippet'],
                "content": content
            }
            visited_links.append({"title": item["title"], "link": item['link']})
            del content
            skeemed_search_results.append(formatted_content)
            del formatted_content
        except Exception as e:
            formatted_content = {
                "title": item['title'],
                "link": item['link'],
                "snippet": item['snippet'],
                "content": "cannot process data from link"
            }
            skeemed_search_results.append(formatted_content)
            blocked_links.append({"title": item['title'], "link": item['link']})
            del formatted_content
            continue

    return skeemed_search_results, visited_links, blocked_links


def custom_deeper_google_research_agent(query, google_custom_search_api_key, google_custom_search_engine_id, llm, llm_client_api_key) -> dict:
    """
    Ask the internet
    # google
    # ms markitdown for crawl
    # groq for summarise
    Args:
        query (str): 

    Returns:
        str:
    """
    response = None
    print("searching google")
    search_results = google_custom_search(
        query, google_custom_search_api_key, google_custom_search_engine_id)
    # Get all links
    snippets = list(map(lambda item: {
                    "title": item['title'], "link": item['link'], "snippet": item['snippet']}, search_results))
    del search_results
    skeemed_search_results, visited_links, blocked_links = skeem_search_results(query=query, snippets=snippets, n=5)
    del snippets
    print("searching and skeeming finished")
    # TODO: add this to backup memory
    
    # split 'skeemed_search_results' into a few parts because it's huge as context
    # Calculate chunk size (n/4)
    n = len(skeemed_search_results)
    chunk_size = n // 4 if n % 4 == 0 else (n // 4) + 1

    # Split the list
    split_lists_of_skeemed_search_results = split_list(skeemed_search_results, chunk_size)

    # Output the result
    split_responses = []
    print(f"skeeming result split into {n} parts")
    
    for i, sublist_of_skeemed_search_results in enumerate(split_lists_of_skeemed_search_results):
        print(f"Researhing Sublist {i+1}")
        # groq inference for each part
        # on conclusive groq inference again with all the parts as context
        system_prompt = "You are a helpful AI assistant. You are given a google search query and the search results. You need to create a research result, which will be the summary of the given information. Extract the most useful information first. There could be bullte points, lists, tables, etc. Put the most relevant information first. Put links as reference in text if necessary."
        prompt = f"Google search query: {str(query)},\nGoogle search result:\n{json.dumps(sublist_of_skeemed_search_results)}\n\nCreate a research result response from the given information. Put the most relevant information first. Put links as reference in text if necessary."
        
        response, model = groq_inference(
            message=prompt, model=llm, api_key=llm_client_api_key, system_message=system_prompt)

        response = {
            "role": f"groq_assistant-{model}",
            "inference_index": i+1,
            "type": "search_result_summary", 
            "content": response
            }
        split_responses.append(response)
        time.sleep(60)
    
    print(f"researhing {n} parts finished, requesting conclusive response from groq")
    
    system_prompt = "You are a helpful AI assistant. You are given a list of reponses content from your research. Write a summary of the given information. Extract the most useful information first. There could be bullte points, lists, tables, etc. Put the most relevant information first. Put links as reference in text if necessary."
    prompt = f"Google search query: {str(query)},\nNumber of inference in research:{len(sublist_of_skeemed_search_results)}\nResearch result:\n{json.dumps(split_responses)}\n\nCreate a research result from the given information. Put the most relevant information first. Put links as reference in text if necessary. Be concise and give small reply. Do not write too much text."
    response, model = groq_inference(
            message=prompt, model=llm, api_key=llm_client_api_key, system_message=system_prompt)
    del system_prompt
    del prompt
    del split_responses
    del split_lists_of_skeemed_search_results
    del skeemed_search_results
    print(f"got response from groq")
    # TODO: add this to backup memory
    
    result = {
        "query": query,
        "resurch_llm_api": "groq",
        "crawler": "ms_markitdown",
        "content": response,
        "type": "custom_deeper_google_research_agent",
        "visited_links": visited_links,     # TODO: issue, getting []
        "crawl_blocked_links": blocked_links,       # TODO: check
        "timestamp": datetime.now().isoformat()
    }
    return result

# test


"""
def test_custom_deeper_google_research_agent():
    print("starting test")
    config = load_config()
    query = "What are the current states of quantum computing research in the world?"
    api_key = config.get("google_custom_search_api_key")
    custom_search_engine_id = config.get("google_custom_search_engine_id")
    result = custom_deeper_google_research_agent(
        query=query,
        google_custom_search_api_key=api_key,
        google_custom_search_engine_id=custom_search_engine_id,
        llm=config.get("groq_model_name"),
        llm_client_api_key=config.get("groq_api_key")
        )
    print("ustom_deeper_google_research_agent response: ", str(result))
    del result


start_time = time.time()
tracemalloc.start()
test_custom_deeper_google_research_agent()
end_time = time.time() 
# Calculate propagation delay
propagation_delay = end_time - start_time
print(f"Propagation delay: {propagation_delay:.6f} seconds")
"""


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
