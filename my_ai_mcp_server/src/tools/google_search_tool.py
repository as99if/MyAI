from asyncio import sleep
from datetime import datetime
import json
import os
import pprint
import tempfile
import time
from typing import Any
import webbrowser
import markdown
import requests
import random
from src.tools.groq import groq_inference
from src.tools.ms_markitdown import generate_markdown_crawled_and_summarize
# from src.core.core_utils import display_notification_with_button
from my_ai.src.utils.my_ai_utils import load_config, split_list
from urllib.parse import urlencode
import tracemalloc
from src.config.config import api_keys

# TODO: manage keys and ids vars from config and env

def embed_search_result(query: str):
    from urllib.parse import quote_plus
    # Encode the search query
    config = load_config()
    encoded_query = quote_plus(query)
    url = f"{config.get('google_custom_search_engine_url')}{encoded_query}"
    # "https://cse.google.com/cse?cx={search_engine_id}%23gsc.tab=0&gsc.q=Rust%20Programming"

    webbrowser.open(url)

    return


def google_custom_search(query, api_keys.google_custom_search_api_key, google_custom_search_engine_id, num_results=10) -> dict:
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


def skeem_search_results(query, snippets, n: int = 10) -> list[dict]:
    """
    - Take first n links from search results
    - Get remaining links after n
    - Randomly select 3 from remaining if available
    - Process selected links to get content
    - Crawl all inner links in the selected links
    - Summarize the contents
    - Get combined results of all links

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
            print(f"Visiting page - {item['title']} - {item['link']}")
            
            content = generate_markdown_crawled_and_summarize(
                main_prompt=query,
                file_path=item['link'],
                if_ingest=False,
                if_crawl=True
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
            print(f"Skimming complete - {item['title']} - {item['link']}")
            
            del formatted_content
        except Exception as e:
            formatted_content = {
                "title": item['title'],
                "link": item['link'],
                "snippet": item['snippet'],
                "content": "cannot process data from link"
            }
            print(f"Skimming failed - {item['title']} - {item['link']}")
            
            skeemed_search_results.append(formatted_content)
            blocked_links.append({"title": item['title'], "link": item['link']})
            del formatted_content
            continue
    for link in random_links:
        try:
            print(f"Visiting page - {item['title']} - {item['link']}")
            
            content = generate_markdown_crawled_and_summarize(
                main_prompt=query,
                file_path=item['link'],
                if_ingest=False,
                if_crawl=True
            )
            formatted_content = {
                "title": item['title'],
                "link": item['link'],
                "snippet": item['snippet'],
                "content": content
            }
            
            
            visited_links.append({"title": item["title"], "link": item['link']})
            print(f"Skimming complete - {item['title']} - {item['link']}")
            
            del content
            skeemed_search_results.append(formatted_content)
            del formatted_content
        except Exception as e:
            formatted_content = {
                "title": item['title'],
                "link": item['link'],
                "snippet": item['snippet'],
                "content": "cannot process data from link",
            }
            skeemed_search_results.append(formatted_content)
            print(f"Skimming failed - {item['title']} - {item['link']}")
            blocked_links.append({"title": item['title'], "link": item['link']})
            del formatted_content
            continue

    return skeemed_search_results, visited_links, blocked_links

markdown_response = """
        ## Sample
    """

def custom_deeper_google_research_tool(query, google_custom_search_api_key, google_custom_search_engine_id, llm, llm_client_api_key) -> dict:
    """
    Ask the internet
    # google custom search
    # ms markitdown for crawl
    # groq for summarisation
    Args:
        query (str): 

    Returns:
        str:
    """
    planner_memory = []
    
    system_prompt = "You are a helpful AI assistant. You are given a list of reponses content from your research. Write a summary of the given information. Extract the most useful information first. There could be bullte points, lists, tables, etc. Put the most relevant information first. Put links as reference in text if necessary."
    
    # deep research with planned steps
    prompt_ds = f"You have Think or create the steps to research deeply on the topic - {query}"
    prompt_ds_ = {
            "role": f"assistant",
            "type": "groq_research_planning_request", 
            "content": prompt_ds,
            "timestamp": datetime.now().isoformat()
        }
    planner_memory.append(prompt_ds_)
    # TODO add this to  backup memory
    
    response, model = groq_inference(
            message=prompt_ds_, api_key=llm_client_api_key, system_message=system_prompt, task_memory_messages=[])
    response_ds = {
        "query": query,
        "role": f"groq_assistant-{model}",
        "content": response,
        "type": "groq_research_planner_response",
        "timestamp": datetime.now().isoformat()
    }
    planner_memory.append(response_ds)
    # TODO add this to  backup memory
    
    
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
    task_memory = []
    # TODO to add all inference call in memeory
    for i, sublist_of_skeemed_search_results in enumerate(split_lists_of_skeemed_search_results):
        print(f"Researhing Sublist {i+1}")
        # groq inference for each part
        # on conclusive groq inference again with all the parts as context
        system_prompt = "You are a helpful AI assistant. You are given a google search query and the search results. You need to create a research result, which will be the summary of the given information. Extract the most useful information first. There could be bullte points, lists, tables, etc. Put the most relevant information first. Put links as reference in text if necessary."
        prompt = f"Google search query: {str(query)},\nGoogle search result:\n{json.dumps(sublist_of_skeemed_search_results)}\n\nCreate a research result response from the given information. Put the most relevant information first. Put links as reference in text if necessary."
        prompt_ = {
            "role": f"assistant",
            "type": "groq_google_research_request", 
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        }
        # TODO: save prompt_ to  backup memory 
        task_memory.append(prompt_)
        response, model = groq_inference(
            message=prompt, model=llm, api_key=llm_client_api_key, system_message=system_prompt)

        response = {
            "role": f"groq_assistant-{model}",
            "inference_index": i+1,
            "type": "search_result_summary", 
            "content": response,
            "timestamp": datetime.now().isoformat()
            }
        # TODO: save to  backup memory response
        task_memory.append(response)
        
        
        split_responses.append(response)
        time.sleep(60)
    
    print(f"researhing {n} parts finished, requesting conclusive response from groq")
    
    system_prompt = "You are a helpful AI assistant. You are given a list of reponses content from your research. Write a summary of the given information, in markdown format. Extract the most useful information first. There could be bullte points, lists, tables, etc. Put the most relevant information first. Put links as reference in text if necessary."
    prompt = f"Google search query: {str(query)},\nNumber of inference in research:{len(sublist_of_skeemed_search_results)}\nResearch result:\n{json.dumps(split_responses)}\n\nCreate a research result from the given information in Markdown format. Put the most relevant information first. Put links as reference in text if necessary. Be concise and give small reply. Do not write too much text."
    
    response, model = groq_inference(
            message=prompt, model=llm, api_key=llm_client_api_key, system_message=system_prompt, task_memory_messages=planner_memory)
    markdown_response = f"""
       {str(response)}
    """
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
        "role": f"groq_assistant-{model}",
        "type": "groq_google_research_response",
        "visited_links": visited_links,     # TODO: issue, getting []
        "crawl_blocked_links": blocked_links,       # TODO: check
        "timestamp": datetime.now().isoformat()
    }
    # TODO: save to memory result
    
    
    # notification
    # display_notification_with_button(
    #     title="Groq-Google Research Completed",
    #     subtitle=query,
    #     message="Google research details stored in memory."
    #     buttons=[
    #         "Open"
    #     ],
    #     button_actions=[
    #         embed_markdown_to_browser
    #     ]
    # )
    return result


def embed_markdown_to_browser(markdown_string=markdown_response, browser='default'):
    """Renders a Markdown string as HTML and displays it in a web browser.

    Args:
        markdown_string: The Markdown string to render.
        browser:  Specifies the browser to use.  'default' uses the system's
                  default browser.  You can also specify browser names as
                  described in the webbrowser module (e.g., 'firefox', 'chrome').
    """
    # Convert Markdown to HTML
    html_content = markdown.markdown(markdown_string)

    # Create a temporary HTML file
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode='w', encoding='utf-8') as f:
        filepath = f.name
        f.write(html_content)

    # Open the HTML file in the web browser
    webbrowser.get(browser).open('file://' + os.path.realpath(filepath))

# test


"""def test_google_custom_search():
    config = load_config()
    query = "what is the current temparature of Stuttgart?"
    api_key = api_keys.google_custom_search_api_key
    custom_search_engine_id = config.get("google_custom_search_engine_id")
    items = google_custom_search(query=query, google_custom_search_api_key=api_key, google_custom_search_engine_id=custom_search_engine_id)
    pprint.pprint(items, indent=2)

test_google_custom_search()"""

# test


"""
def test_custom_deeper_google_research_agent():
    print("starting test")
    config = load_config()
    query = "What are the current states of quantum computing research in the world?"
    api_key = api_keys.google_custom_search_api_key
    custom_search_engine_id = config.get("google_custom_search_engine_id")
    result = custom_deeper_google_research_agent(
        query=query,
        google_custom_search_api_key=api_key,
        google_custom_search_engine_id=custom_search_engine_id,
        llm=config.get("groq_model_name"),
        llm_client_api_key=api_keys.groq_api_key
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
