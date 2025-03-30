
"""_summary_

- custom_search_agent
"""
from asyncio import sleep

import pprint
from typing import Any
import urllib.parse

import webbrowser
from my_ai.src.utils.my_ai_utils import load_config
import googlemaps
from datetime import datetime
from src.config.config import api_keys

def google_maps_client(location):
    config = load_config()
    gmaps = googlemaps.Client(key=api_keys.google_maps_api_key)

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
    # have to use maps javascript api
    
    print("embed maps")
    config = load_config()
    encoded_location = urllib.parse.quote(location)
    iframe_src = f"https://www.google.com/maps/embed/v1/place?key={api_keys.google_maps_api_key}&q={encoded_location}"

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
    
    # HTML content with the Google Maps iframe
    iframe_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Google Map</title>
    </head>
    <body>
        <h1>Google Map Embed</h1>
        <iframe
            id="embed-map"
            width="600"
            height="450"
            style="border:0"
            loading="lazy"
            allowfullscreen
            referrerpolicy="no-referrer-when-downgrade"
            src="{iframe_src}">
        </iframe>
    </body>
    </html>
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


def test_custom_search_agent():
    pass

def test_maps(location):
    pprint.pprint(google_maps_client(location), indent=2)

location = "Eiffel Tower, Paris"
test_maps(location)
embed_maps(location)