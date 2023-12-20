import json

with open('./resources/aa.json', 'r') as f:
    for item in json.load(f)['items']:
        print(f"{item['pl_hostname']}")  #, item['planet_type'])
