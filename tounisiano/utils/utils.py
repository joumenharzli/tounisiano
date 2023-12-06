import requests
import tempfile


def download_file(url):
    destination = tempfile.NamedTemporaryFile(delete=False).name
    response = requests.get(url)
    with open(destination, "wb") as file:
        file.write(response.content)
    return destination
