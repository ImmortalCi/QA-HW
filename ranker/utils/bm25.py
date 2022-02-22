import torch
import argparse
import pandas as pd
from tqdm import tqdm
import jieba
from ranker.utils.metric import PrecisionAtNum


class BM25(object):

    def __init__(self, docs):
        self._D = len(docs)
        self._dl = torch.tensor([len(doc) for doc in docs])
        self._avg_dl = self._dl.sum().float() / self._D
        self._words = set([word.lower() for doc in docs for word in doc])
        self._word2id = {word: i for i, word in enumerate(self._words)}
        self._n_words = len(self._words)

        f = []
        for doc in docs:
            word_ids = torch.tensor([self._word2id[word.lower()] for word in doc])
            count = (word_ids.unsqueeze(1) == torch.arange(self._n_words).unsqueeze(0)).sum(0)
            f.append(count)
        self._f = torch.stack(f).float()
        self._df = (self._f > 0).sum(dim=0)

        self._idf = torch.log((self._D - self._df + 0.5) / (self._df + 0.5))
        self._k1 = 1.2
        self._b = 0.75

        # to save memory
        del self._words
        del self._df

    def simall(self, doc):
        """

        :param doc(List[str]):
            分词后的query文档
        :return:
            generator(str):
                按分值从小到大生成文档
        """

        word_ids = torch.tensor([self._word2id.get(word.lower(), -1) for word in doc])
        mask = word_ids < 0
        fi = self._f[:, word_ids].transpose(0, 1)
        fi[mask] = 0

        k = self._k1 * (1 - self._b + self._b * self._dl / self._avg_dl)
        r = (fi * (self._k1 + 1)) / (fi + k)
        idf = self._idf[word_ids]

        scores = idf.unsqueeze(1) * r
        scores = scores.sum(0)
        scores, indices = torch.sort(scores, descending=True)
        return (i for i in indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BM25'
    )
    parser.add_argument('--train_file', default='data/answer.csv',
                        help='train file')
    parser.add_argument('--test_file', default='data/test_candidates.txt',
                        help='test file')
    parser.add_argument('--question', default='data/question.csv',
                        help='all questions')
    parser.add_argument('--candidates_num', default='30',
                        help='the number of candidates')

    args, _ = parser.parse_known_args()

    # 生成答案列表
    answer_data = pd.read_csv(args.train_file, index_col='ans_id')
    answer_list = []
    for i in range(answer_data.shape[0]):
        answer_list.append(answer_data.loc[i, 'content'])

    # 读取测试集
    query2answer_id = dict()
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_list = f.readlines()
        for i in test_list[1:]:
            test_pair = tuple(int(x) for x in (i.strip().split(',')))
            if test_pair[3] == 1:
                if test_pair[0] not in query2answer_id.keys():
                    query2answer_id[test_pair[0]] = [test_pair[1]]
                else:
                    query2answer_id[test_pair[0]].append(test_pair[1])

    # 读取问题
    id2question = dict()
    question_dataframe = pd.read_csv(args.question)
    for i in range(question_dataframe.shape[0]):
        id2question[question_dataframe.loc[i, 'question_id']] = question_dataframe.loc[i, 'content']

    s = BM25(answer_list)
    all_query_ids = []
    all_answers_ids = []
    for query_id, answers_id in query2answer_id.items():
        all_query_ids.append(query_id)
        all_answers_ids.append(answers_id)
    assert len(all_query_ids) == len(all_answers_ids)
    result = []  # 返回的就是candidates答案的索引
    for query_id in all_query_ids:
        result.append(s.simall(list(jieba.cut(id2question[query_id]))))

    labels = []
    for pred, gold in tqdm(zip(result, all_answers_ids)):
        label = []
        for p in pred:
            if len(label) == args.candidates_num:
                break
            elif p in gold:
                label.append(1)
            else:
                label.append(0)
        labels.append(label)

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
