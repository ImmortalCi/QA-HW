import json
from collections import Counter
from tqdm import tqdm

pairs = []
with open("/Users/immortalci/PycharmProjects/QA-HW/data/train_pair_postive.json", "r", encoding="utf-8") as f:
    for line in f:
        line = json.loads(line.strip())
        s1, s2 = line['query'], line['title']
        if s1 and s2:
            pairs.append((s1, s2))

sentences = []
for x in pairs:
    sentences.append(x[0])
    sentences.append(x[1])
s2freq = Counter(sentences)

s12s2 = {}
for x in pairs:
    if x[0] not in s12s2:
        s12s2[x[0]] = [x[1]]
    else:
        s12s2[x[0]].append(x[1])

s22s1 = {}
for x in pairs:
    if x[1] not in s22s1:
        s22s1[x[1]] = [x[0]]
    else:
        s22s1[x[1]].append(x[0])

result = {}
def process(sentence, s12s2, s22s1):
    res = set()
    def _process(sentence, s12s2, s22s1, cur_res):
        if sentence not in s12s2 and sentence not in s22s1:
            return set()

        new_added = set()
        if sentence in s12s2:
            cur_res.add(sentence)
            similar = set(s12s2.pop(sentence))
            for s in similar:
                if s not in cur_res:
                    new_added.add(s)
                    
        if sentence in s22s1:
            cur_res.add(sentence)
            similar = set(s22s1.pop(sentence))
            for s in similar:
                if s not in cur_res:
                    new_added.add(s)
        for s in new_added:
            _process(s, s12s2, s22s1, cur_res)

    _process(sentence, s12s2, s22s1, res)
    return res

def get_standard(res, s2freq):
    res = list(res)
    res = list(sorted(res, key= lambda x: s2freq[x], reverse=True))
    standard = res[0]
    extends = res[1:]
    return standard, extends

count = 0
sentences = sorted(list(set(sentences)))
for s in tqdm(sentences):
    try:
        res = process(s, s12s2, s22s1)
    except Exception as e:
        count += 1
        continue
    if len(res) > 0 and len(res) <= 20:
        standard, extends = get_standard(res, s2freq)
        assert standard not in result
        result[standard] = extends
print(count)
print(len(s12s2), len(s22s1))
with open("train.json","w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)