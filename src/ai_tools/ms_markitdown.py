from xml.etree import ElementTree
from groq import Groq
from markitdown import MarkItDown
import requests

from src.ai_tools.groq import groq_inference
from src.utils.utils import load_config


def generate_markdown_crawled_and_summarize(query, url, if_crawl: bool = False, if_ingest: bool = False):
    
    results = []
    results_ = []
    
    config = load_config()
    md = MarkItDown()
    # md = MarkItDown(llm_client=Groq(api_key=config.get("groq_api_key")), llm_model=config.get("groq_model_name"))
    result = md.convert(link)
    system_prompt = "You are a helpful AI assistant to summarize markdown text."
    prompt = f"Main prompt:{query}\nMarkdown Text:\n{str(result.text_content)}\n\nSummarize the text for research. Be  concise, write in bullete points if necessary. And, only wirte the most important information given the main prompt."
    response = groq_inference(
        message=prompt, model=config.get("groq_model_name"), api_key=Groq(api_key=config.get("groq_api_key")), system_message=system_prompt)
    results_.append({"link": link,
                    "content": result, "summary": response})
    results.append({"link": link, "summary": response})

    if if_crawl:
        urls = []
        try:
            sitemap_url = url + "/sitemap.xml"
            urls = get_sitemap_urls(sitemap_url)
        except Exception as e:
            print(f"Error getting urls from sitemap of {url}: ", e)
            
        crwaled_pages = 0
        for link in urls:
            try:
                result = md.convert(link)
                crwaled_pages = crwaled_pages + 1
                system_prompt = "You are a helpful AI assistant to summarize markdown text."
                prompt = f"Main prompt:{query}\nMarkdown Text:\n{str(result.text_content)}\n\nSummarize the text for research. Be  concise, write in bullete points if necessary. And, only wirte the most important information given the main prompt."
                response = groq_inference(
                    message=prompt, model=config.get("groq_model_name"), api_key=Groq(api_key=config.get("groq_api_key")), system_message=system_prompt)
                results_.append({"link": link, "root_link": url,
                                "content": result, "summary": response})
                results.append({"link": link, "root_link": url, "summary": response})

            except Exception as e:
                print(f"Error during crawling or summarizing {link}:", e)
                results_.append({"link": link, "root_link": url,
                                "content": "Unable to crawl."})
                continue

    response = {
        "query": query,
        "link": url,
        "crwaled_pages": crwaled_pages,
        "content": results
    }
    if if_ingest:
        for item in results_:
            pass
            # TODO: ingest(item)

    return response


def generate_markdown_and_summarize(main_prompt, file_path, if_ingest: bool = False):
    config = load_config()
    # md = MarkItDown(llm_client=Groq(api_key=config.get("groq_api_key")), llm_model=config.get("groq_model_name"))
    md = MarkItDown()
    result = md.convert(file_path)
    system_prompt = "You are a helpful AI assistant to summarize markdown text."
    prompt = f"Main prompt:{main_prompt}\nMarkdown Text:\n{str(result.text_content)}\n\nSummarize the text for research. Be  concise, write in bullete points if necessary. And, only wirte the most important information given the main prompt."
    response = groq_inference(
        message=prompt, model=config.get("groq_model_name"), api_key=Groq(api_key=config.get("groq_api_key")), system_message=system_prompt)
    del md
    if if_ingest:
        pass
        # TODO: ingest({"link": file_path, "content": result, "summary": response})

    return response


def get_markdown_content(path, if_ingest: bool = False):
    md = MarkItDown()
    result = md.convert(path)
    print(result)
    del md
    if if_ingest:
        pass
        # TODO: ingest({"link": file_path, "content": result, "summary": response})
    return result.text_content

# https://github.com/as99if/ottomator-agents/blob/main/crawl4AI-agent/crawl4AI-examples/3-crawl_docs_FAST.py
def get_sitemap_urls(url):
    """
    Fetches all URLs from the Pydantic AI documentation.
    Uses the sitemap (https://ai.pydantic.dev/sitemap.xml) to get these URLs.

    Returns:
        List[str]: List of URLs
    """
    # sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()

        # Parse the XML
        root = ElementTree.fromstring(response.content)

        # Extract all URLs from the sitemap
        # The namespace is usually defined in the root element
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]

        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []


# test
"""
def test_get_markdown_content():
    path = "https://stackoverflow.com/questions/3437059/does-python-have-a-string-contains-substring-method"
    r = get_markdown_content(path)
    print(r)
    
test_get_markdown_content()
"""
