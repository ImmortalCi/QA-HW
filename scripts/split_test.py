import json
import random
random.seed(1)

train_file = "data/total.json"

with open(train_file, "r", encoding="utf-8") as f:
    total = json.load(f)

def split(data, ratio=0.2):
    train, test = {}, {}
    for standard, extends in data.items():
        length = len(extends)
        tlen = round(length * ratio)
        index = list(range(length))
        test_index = random.sample(index, k=tlen)
        train_index = [i for i in index if i not in test_index]
        test_extend = [extends[i] for i in test_index]
        train_extend = [extends[i] for i in train_index]
        train[standard] = train_extend
        if len(test_extend) > 0:
            test[standard] = test_extend
    return train, test

train, test = split(total)

with open("data/train.json", 'w', encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=4)

with open("data/test.json", 'w', encoding="utf-8") as f:
    json.dump(test, f, ensure_ascii=False, indent=4)