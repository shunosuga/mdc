import requests
import json

# Define base URL for datasets API
base_url = "https://api.semanticscholar.org/datasets/v1/release/"

# To get the list of available releases make a request to the base url. No additional parameters needed.
response = requests.get(base_url)

# Print the response data
print(response.json())

base_url = "https://api.semanticscholar.org/datasets/v1/release/"

# Set the release id
release_id = "2025-08-12"

# Make a request to get datasets available the latest release
response = requests.get(base_url + release_id)

# Print the response data
print(response.json())

base_url = "https://api.semanticscholar.org/datasets/v1/release/"

# This endpoint requires authentication via api key
api_key = "i12KtWkXux3OIh0nqaLEC12coSa4Pa7N1ZJREMCq"
headers = {"x-api-key": api_key}

# Set the release id
release_id = "2025-08-12"

# Define dataset name you want to download
dataset_name = 's2orc_v2'

# Send the GET request and store the response in a variable
response = requests.get(base_url + release_id + '/dataset/' + dataset_name, headers=headers)

# Process and print the response data
with open('response.json', 'w', encoding='utf-8') as f:
    json.dump(response.json(), f, ensure_ascii=False, indent=2)