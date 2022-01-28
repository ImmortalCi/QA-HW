import json
import random
from collections import namedtuple

QueryPair = namedtuple(typename='QueryPair', field_names=['standard', 'extend'])
Triplet = namedtuple(typename='triplet', field_names=['user_query', 'positive_query', 'negative_querys'])

class Corpus(object):
    def __init__(self, pairs):
        super(Corpus, self).__init__()
        self._pairs = pairs

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, index):
        return self._pairs[index]

    def __add__(self, other):
        pairs = self._pairs + other._pairs
        return Corpus(pairs)

    @property
    def standards(self):
        return [pair.standard for pair in self]

    @property
    def extends(self):
        return [extend for pair in self for extend in pair.extend]

    @classmethod
    def load(cls, fname):
        pairs = []
        with open(fname, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for key, value in data.items():
            pairs.append(QueryPair(key,value))

        corpus = cls(pairs)

        return corpus

    def sample(self, percent=1):
        result=[]
        for pair in self:
            standard = pair.standard
            extend = pair.extend
            length = round(len(extend) * percent)
            chosen = set(random.sample(range(len(extend)), k=length))
            new_extend = [extend[i] for i in chosen]
            result.append(QueryPair(standard, new_extend))
        return Corpus(result)

class TrainingSamples(object):
    def __init__(self, triplets):
        super(TrainingSamples, self).__init__()
        self._triplets = triplets

    def __len__(self):
        return len(self._triplets)

    def __getitem__(self, index):
        return self._triplets[index]

    def __add__(self, other):
        pairs = self._triplets + other._pairs
        return TrainingSamples(pairs)

    @property
    def user_querys(self):
        return [t.user_query for t in self]

    @property
    def positive_querys(self):
        return [t.positive_query for t in self]

    @property
    def negative_querys(self):
        return [t.negative_querys for t in self]

    @classmethod
    def from_corpus(cls, corpus, k):
        triplet = []
        total = sorted(set(corpus.standards) | set(corpus.extends))
        for pair in corpus:
            positive_cluster = sorted([pair.standard] + pair.extend)
            for extend in pair.extend:
                if set(positive_cluster).issubset(set([extend])):
                    continue
                for _ in range(k):
                    while True:
                        p = random.choice(positive_cluster)
                        if p != extend:
                            break
                    while True:
                        n = random.choice(total)
                        if n not in positive_cluster:
                            break
                    triplet.append(Triplet(extend, p, n))
        return cls(triplet)

    @classmethod
    def from_txt(cls, path):
        triplet = []
        with open(path) as f:
            for line in f:
                data = json.load(line)
                triplet.append(Triplet(data['query'], data['positive'], data['negative']))
        return cls(triplet)

    def sample(self, k):
        triplets = random.sample(self._triplets, k)
        return TrainingSamples(triplets)
