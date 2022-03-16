import json
import random
from collections import namedtuple
import pandas as pd
from tqdm import tqdm

QueryPair = namedtuple(typename='QueryPair', field_names=['standard', 'extend'])
Triplet = namedtuple(typename='triplet', field_names=['user_query', 'positive_query', 'negative_query'])
QueryAnswer = namedtuple(typename='QueryAnswer', field_names=['query', 'answers'])

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

    @classmethod
    def load_BM25(cls, fname):  # 最外层是列表
        pairs = []
        with open(fname, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for dic in data:
            for key, value in dic.items():
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
    def from_corpus(cls, corpus):
        triplet = []
        total = sorted(set(corpus.standards) | set(corpus.extends))
        for pair in corpus:
            positive_cluster = sorted([pair.standard] + pair.extend)
            for extend in pair.extend:
                if set(positive_cluster).issubset(set([extend])):
                    continue
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
    def from_corpus_stage2(cls, corpus):  # 精排模型
        triplet = []
        for pair in corpus:
            query = pair.standard
            neg = []
            all_samples = pair.extend[:20]  # list嵌套list，如果是标签，就多嵌套一层list
            flag = False
            for sample in all_samples:
                if type(sample[1]) == list:  # 说明存在label
                    flag = True
                    break
            if flag:
                for sample in all_samples:
                    if type(sample[1]) == list:
                        pos = sample[1][0]
                    else:
                        neg.append(sample[0])
                for neg_sample in neg:
                    triplet.append(Triplet(query, pos, neg_sample))
            else:
                continue
        return cls(triplet)

    @classmethod
    def from_corpus_stage2_BM25(cls, corpus):
        triplet = []
        for pair in corpus:
            query = pair.standard
            neg = []
            all_samples = pair.extend[:20]  # list
            flag = False
            for sample in all_samples:
                if type(sample) == list:
                    flag = True
                    break
            if flag:
                for sample in all_samples:
                    if type(sample) == list:
                        pos = sample[1]
                    else:
                        neg.append(sample)
                for neg_sample in neg:
                    triplet.append(Triplet(query, pos, neg_sample))
            else:
                continue
        return cls(triplet)




    @classmethod
    def QA_from_corpus(cls, corpus, queryid2str, answerid2str):
        triplet = []
        total_answer = answerid2str
        for pair in tqdm(corpus):
            query_string = queryid2str[pair.query]
            positive_cluster = [string for id in pair.answers for string in answerid2str[id]]
            for answer in positive_cluster:
                p = answer
                while True:
                    n = random.choice(total_answer)
                    if n not in positive_cluster:
                        break
                triplet.append(Triplet(query_string, p, n))
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

class QA_Corpus(object):
    def __init__(self, pairs):
        super(QA_Corpus, self).__init__()
        self._pairs = pairs

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, index):
        return self._pairs[index]

    def __add__(self, other):
        pairs = self._pairs + other._pairs
        return Corpus(pairs)

    @property
    def queries(self):
        return [pair.query for pair in self._pairs]

    @property
    def answers(self):
        return [answers for pair in self._pairs for answers in pair.answers]

    @classmethod
    def load(cls, fname):
        pairs = []
        query2answer_id = dict()
        with open(fname, 'r', encoding='utf-8') as f:
            train_list = f.readlines()
            for i in train_list[1:]:
                train_pair = tuple(int(x) for x in (i.strip().split(',')))
                if train_pair[0] not in query2answer_id.keys():
                    query2answer_id[train_pair[0]] = [train_pair[1]]
                elif train_pair[0] in query2answer_id.keys() and train_pair[1] not in query2answer_id[train_pair[0]]:
                    query2answer_id[train_pair[0]].append(train_pair[1])
                else:
                    continue
        for key, value in query2answer_id.items():
            pairs.append(QueryAnswer(key, value))

        corpus = cls(pairs)

        return corpus

def query_id2str(fname):
    id2question = dict()
    question_dataframe = pd.read_csv(fname)
    for i in range(question_dataframe.shape[0]):
        id2question[question_dataframe.loc[i, 'question_id']] = question_dataframe.loc[i, 'content']
    return id2question

def answer_str(fname):
    answer_data = pd.read_csv(fname, index_col='ans_id')
    answer_list = []
    for i in range(answer_data.shape[0]):
        answer_list.append(answer_data.loc[i, 'content'])

    return answer_list
