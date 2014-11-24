import json


def build_data(filename='labeled.txt'):
    data = []
    with open(filename) as f:
        while True:
            t = f.readline()
            if t is None or t == '':
                break
            d = json.loads(t)
            data.append({'text': d['text'], 'class': d['sentiment']})
    return data
    