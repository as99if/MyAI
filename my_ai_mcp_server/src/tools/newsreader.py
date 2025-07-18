## TODO: refine

from newsapi import NewsApiClient
from datetime import datetime
from src.config.config import api_keys
import pprint

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key=api_keys.news_api_org_api_key)

def get_news_updates(
    topic=None,        # Keyword or phrase to search for
    category=None,     # Category: business, entertainment, general, health, science, sports, technology
    country=None,      # 2-letter country code, e.g., 'us', 'gb', 'de'
    from_date=None,    # Start date in 'YYYY-MM-DD' format
    to_date=None,      # End date in 'YYYY-MM-DD' format
    page_size=20       # Number of results per page (max 100)
):
    # Use 'everything' endpoint for date filtering, 'top-headlines' for country/category
    if from_date or to_date:
        # Use 'everything' endpoint for date filtering
        articles = newsapi.get_everything(
            q=topic,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='publishedAt',
            page_size=page_size
        )
    else:
        # Use 'top-headlines' endpoint for country/category/topic
        articles = newsapi.get_top_headlines(
            q=topic,
            category=category,
            country=country,
            page_size=page_size
        )
    return articles['articles']

# Example usage
if __name__ == "__main__":
    # User-defined filters
    topic = 'travel japan'
    category = 'entertainment'
    country = 'jp'
    from_date = '2025-04-01'
    to_date = '2025-04-23'

    news_list = get_news_updates(
        topic=topic,
        category=category,
        country=country,
        from_date=from_date,
        to_date=to_date,
        page_size=5
    )


    pprint.pprint(news_list)
    for idx, article in enumerate(news_list, 1):
        print(f"{idx}. {article['title']}")
        print(f"   Source: {article['source']['name']}")
        print(f"   Published at: {article['publishedAt']}")
        print(f"   URL: {article['url']}\n")
