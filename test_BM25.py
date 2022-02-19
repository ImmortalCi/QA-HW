import math
import argparse
import json
import jieba
from collections import namedtuple
from ranker.utils.metric import PrecisionAtNum

RankPair = namedtuple(typename='RankPair', field_names=['sentence', 'score'])


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def data2sentences_doc(data: dict):
    # 读取训练集，获取句子
    sentences = []
    for standard, extends in data.items():
        sentences.append(standard)
        for extend in extends:
            sentences.append(extend)
    # 读取句子，生成doc
    doc = []
    for sent in sentences:
        words = list(jieba.cut(sent))
        # words = utils.filter_stop(words)
        doc.append(words)
    return sentences, doc


def get_search_result(test_data, BM25, extend2standard, k=30):
    querys = [extend for _, extends in test_data.items() for extend in extends]
    golds = [standard for standard, extends in test_data.items() for _ in extends]
    print('searching...')
    result = []
    for query in querys:
        result.append(BM25.simall(query, k))
    labels = []
    assert len(result) == len(golds)
    for pred, gold in zip(result, golds):
        label = []
        for p in pred:
            if p.sentence == gold or gold in extend2standard.get(p.sentence, []):
                label.append(1)
            else:
                label.append(0)
        labels.append(label)
    return labels


class BM25(object):

    def __init__(self, sentences, docs):
        self.D = len(docs)
        self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.D
        self.sentences = sentences
        self.docs = docs
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {}  # 存储每个词及出现了该词的文档数量
        self.idf = {}  # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)

    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word] * self.f[index][word] * (self.k1 + 1)
                      / (self.f[index][word] + self.k1 * (1 - self.b + self.b * d
                                                          / self.avgdl)))
        return RankPair(sentences[index], score)

    def simall(self, sentence, k=30):
        doc = list(jieba.cut(sentence))
        rank_result = []
        for index in range(self.D):
            rank_pair = self.sim(doc, index)
            rank_result.append(rank_pair)
        rank_result.sort(key= lambda x:x.score, reverse=True)
        return rank_result[:k]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BM25'
    )
    parser.add_argument('--train_file', default='data/train.json',
                        help='train file')
    parser.add_argument('--test_file', default='data/test.json',
                        help='test file')

    args, _ = parser.parse_known_args()
    train_data = read_json(args.train_file)
    test_data = read_json(args.test_file)
    sentences, doc = data2sentences_doc(train_data)
    s = BM25(sentences, doc)

    extend2standard = {}
    for standard, extends in train_data.items():
        for extend in extends:
            if extend not in extend2standard:
                extend2standard[extend] = [standard]
            else:
                print(
                    f'warning: {extend} appear in more than one standard questions')
                extend2standard[extend].append(standard)

    labels = get_search_result(test_data, s, extend2standard, 30)

    p1, p3 = PrecisionAtNum(1), PrecisionAtNum(3)
    p5, p10 = PrecisionAtNum(5), PrecisionAtNum(10)
    p20, p30 = PrecisionAtNum(20), PrecisionAtNum(30)
    for label in labels:
        p1(label)
        p3(label)
        p5(label)
        p10(label)
        p20(label)
        p30(label)
    print(p1)
    print(p3)
    print(p5)
    print(p10)
    print(p20)
    print(p30)





