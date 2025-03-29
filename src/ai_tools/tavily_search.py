# not good

from tavily import TavilyClient
from src.config.config import api_keys
import pprint

tavily_client = TavilyClient(api_key=api_keys.travily_search_api_key)
response = tavily_client.search("Find the best place to eat Biriyani in Stuttgart?", search_depth="advanced")
# pprint.pprint(response['results'])

urls = []
for result in response['results']:
    urls.append(result['url'])

response = tavily_client.extract(urls=urls)
pprint.pprint(response)