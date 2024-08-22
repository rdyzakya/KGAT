import requests
import time
import re

# Disambiguation baseline
def disambiguation_baseline(item):
    try:
        # If item can be converted to an integer, return it directly
        return int(item)
    except ValueError:
        # If not, proceed with the Wikidata search
        try:
            url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
            data = requests.get(url).json()
            # Return the first id (Could upgrade this in the future)
            return data['search'][0]['id']
        except:
            return item

def get_wikidata_id(query):
    search_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": query
    }
    
    response = requests.get(search_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if 'search' in data:
            if len(data['search']) > 0:
                return data['search'][0]['id']
    return None

def my_disambiguation(input_str):
    # Check if the string is an integer
    try:
        return int(input_str)
    except ValueError:
        pass

    # If not an integer, try to get the Wikidata ID
    while True:
        try:
            wikidata_id = get_wikidata_id(input_str)
            break
        except requests.exceptions.ConnectionError as e:
            print("Get wikidata id request timeout, redo request...")
            time.sleep(10)
    if wikidata_id:
        return wikidata_id
    
    # If all else fails, return the original string
    return input_str

def get_wikidata_entity_name(entity_id):
    entity_id = str(entity_id)
    if not re.match(r"Q\d+", entity_id):
        return entity_id
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "format": "json",
        "languages": "en"
    }
    
    response = requests.get(url, params=params)
    data = response.json()

    if 'entities' in data and entity_id in data['entities']:
        entity = data['entities'][entity_id]
        if 'labels' in entity and 'en' in entity['labels']:
            return entity['labels']['en']['value']
    return None