import json

with open('./resources/star_names.txt', 'r') as f:
    for item in sorted(set(f.readlines())):
        print(f"{item.strip()}")  #, item['planet_type'])
