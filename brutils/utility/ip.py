import requests
import json


def ip_geolocate(ip_address):
    request_url = 'https://geolocation-db.com/jsonp/' + ip_address
    response = requests.get(request_url)
    result = response.content.decode()
    result = result.split("(")[1].strip(")")
    result = json.loads(result)
    return result
