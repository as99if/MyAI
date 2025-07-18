

import requests
from langchain.tools import Tool
from src.utils.schemas import ToolCallResponse


def get_current_time(location: str) -> ToolCallResponse:
    """
    Fetches the current time for a given location using the World Time API.
    Args:
        location: The location for which to fetch the current time, formatted as 'Country/Region/City'.
    Returns:
        str: A string indicating the current time in the specified location, or an error message if the request fails.
    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP request.
    """
    url = f"http://worldtimeapi.org/api/timezone/{location}.json"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        current_time = data["datetime"]
 
        return ToolCallResponse(
            tool_used="get_current_time",
            description="Fetches the current time for a given location using the World Time API.",
            result=f"The current time in {location} is {current_time}."
        )

    except requests.exceptions.RequestException as e:
        return f"Error fetching time data: {str(e)}"



def search_wikipedia(query: str) -> ToolCallResponse:
    """
    Fetches a summary from a Wikipedia page for a given query.
    Args:
        query: The search term to look up on Wikipedia.
    Returns:
        str: A summary of the Wikipedia page if successful, or an error message if the request fails.
    Raises:
        requests.exceptions.RequestException: If there is an issue with the HTTP request.
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()

        return ToolCallResponse(
                tool_used="search_wikipedia",
                description="Fetches a summary of a Wikipedia page for a given query.",
                result=f"Summary for {data["title"]}: {data["extract"]}"
            )

    except requests.exceptions.RequestException as e:
        return f"Error fetching Wikipedia data: {str(e)}"