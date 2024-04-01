import requests

# Define the URL of the Flask API endpoint
url = 'http://127.0.0.1:5000/annotate_image'

# Define the JSON payload containing the image path
payload = {"image_path": "./img1.jpg"}

# Send a POST request with the JSON payload
response = requests.post(url, json=payload)

# Check the response status code
if response.status_code == 200:
    # If the request was successful, print the response content
    print(response.json())
else:
    # If there was an error, print the error message
    print(f"Error: {response.text}")
