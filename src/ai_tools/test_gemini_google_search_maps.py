import pprint
from src.ai_tools.gemini_google_search_maps import embed_maps, google_custom_search, google_maps_client
from src.utils.utils import load_config


def test_google_custom_search():
    # REPLACE with your actual values
    config = load_config()

    results = google_custom_search("Rust Programming", api_key=config.get('google_custopm_search_api_key'), custom_search_engine_id=config.get('google_custom_search_engine_id'))
    print(results)

# test_google_custom_search()

# python -m src.ai_tools.test_gemini_google_search_maps

def test_custom_search_agent():
    pass

def test_maps(location):
    pprint.pprint(google_maps_client(location), indent=2)

location = "Eiffel Tower, Paris"
test_maps(location)
embed_maps(location)