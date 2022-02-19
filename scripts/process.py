import json
filename = 'data/raw.json'

result = {}
with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        result[data['text']] = data['synonyms']

with open('data/train.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)