import pandas as pd
import requests
import json

# Define the API endpoint and parameters
url = 'https://api.patentsview.org/patents/query'

# Define the query as a JSON object
query = {
    "_and": [
        {"grant_date": {"_gte": "2020-01-01"}},
        {"grant_date": {"_lte": "2020-12-31"}}
    ]
}

# Send a GET request to the API with the JSON query
response = requests.get(url, json=query)

# Parse the JSON response
data = json.loads(response.text)

# Print the API response
print(data)