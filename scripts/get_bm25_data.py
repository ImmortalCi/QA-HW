import argparse
import json
from multiprocessing import Manager, Process

from elasticsearch import Elasticsearch
from tqdm import tqdm
import jieba

import sys

sys.path.append(".")
from ranker.utils.metric import PrecisionAtNum


def tokenize(text):
    text = text.replace('\t', ' ')
    text = text.replace('\"', '\'')
    text = text.replace('\n', ' ')
    words = []
    for token in jieba.cut(text):
        if not token.isspace():
            words.append(token)
    if len(words) == 0:
        return [text]

    return words


def getBM25Result(segmentQuestion, es, index, size):
    dsl = {
        "query": {
            "bool": {
                "must": [
                    {
                        "nested": {
                            "query": {
                                "match": {
                                    "ex_questions.content_after_pre_tokenizer": {
                                        "query": segmentQuestion,
                                        "operator": "OR",
                                        "prefix_length": 0,
                                        "max_expansions": 100,
                                        "fuzzy_transposition": True,
                                        "lenient": False,
                                        "zero_terms, query": "NONE",
                                        "auto_generate_synonyms_phrase_query": True,
                                        "boost": 1.0
                                    }
                                }
                            },
                            "path": "ex_questions",
                            "ignore_unmapped": False,
                            "score_mode": "max",
                            "boost": 1.0,
                            "inner_hits": {
                                "ignore_unmapped": False,
                                "from": 0,
                                "size": 30,
                                "version": False,
                                "explain": False,
                                "track_scores": False,
                                "_source": False,
                                "docvalue_fields": [
                                    "ex_questions.content.raw_content"
                                ]
                            }
                        }
                    }
                ],
                "adjust_pure_negative": True,
                "boost": 1.0
            }
        }
    }
    res = es.search(index=index, body=dsl, from_=0, size=size)
    return res['hits']['hits']


def getHits(hits):
    hits = hits['ex_questions']['hits']['hits']
    question2score = {}
    for questionElement in hits:
        question = questionElement['fields']['ex_questions.content.raw_content'][0]
        score = questionElement['_score']
        question2score[question] = score
    return question2score


def bm25ResultToTrainStyle(results):
    sourceQ = results['_source']
    hits = results['inner_hits']

    trainStyle = {}
    trainStyle['max_score'] = hits['ex_questions']['hits']['max_score']
    trainStyle['standard'] = sourceQ['st_question']
    trainStyle['extends'] = []
    for originInfo in sourceQ['ex_questions']:
        questionInfo = originInfo
        content = originInfo['content']
        question2score = getHits(hits)
        if content in question2score:
            questionInfo['score'] = question2score[content]
            if questionInfo[content] == trainStyle['max_score']:
                trainStyle['top_score_question'] = content
        questionInfo.pop("content_after_pre_tokenizer")
        trainStyle['extends'].append(questionInfo)
    return trainStyle


def split(data, n):
    length = len(data)
    list_data = [{'standard': standard, 'extends': extends} for standard, extends in data.items()]
    chunk = length // n + 1
    return [list_data[i: i + chunk] for i in range(0, length, chunk)]


def main(data, es, index, size, n, manager):
    total = []
    for item in tqdm(data):
        standard, extends = item['standard'], item['extends']
        for extend in extends:
            result = {}
            segmented = ','.join(tokenize(extend))
            result['query'] = extend
            result['standard'] = standard
            result['candidates'] = []
            result['labels'] = []
            bm25result = getBM25Result(segmented, es, index, size)
            for item in bm25result:
                qa_pair = bm25ResultToTrainStyle(item)
                for ex in qa_pair['extends']:
                    if 'score' not in ex:
                        ex['score'] = 0
                result['candidates'].append(qa_pair)
                result['labels'].append(qa_pair['standard'] == standard)
            total.append(result)
    manager[n] = total
    print(f'process {n} finished')


def post_process(result):
    def get_max_score(candi):
        return sorted(candi['extends'], key=lambda x: x["score"], reverse=True)[0]['content']

    def get_second_max_score(candi, query):
        for x in sorted(candi['extends'], key=lambda x: x["score"], reverse=True):
            if x != query:
                return x['content']
        print("no positive chosen!!!")

    labels = []
    new_result = []
    for item in result:
        labels.append(item['labels'])
        candidates = []
        for candi in item['candidates']:
            if candi['standard'] == item['standard']:
                second_score = get_second_max_score(candi, item['query'])
                if second_score:  #如果可以找到同类下的其他问
                    candidates.append(['labels', second_score])
                else:  # 找不到同类下的其他问
                    pass
            else:
                candidates.append(get_max_score(candi))
        new_result.append({item['query']: candidates})
    return new_result, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='import data to es'
    )
    parser.add_argument('--train_file', default='data/simCLUE_test.json',
                        help='train file')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=9200)
    parser.add_argument('--index', default='text-recall-es-hwnlp')
    parser.add_argument('--size', type=int, default=20)
    parser.add_argument('--out_path', default='/data1/wtc/HW-QA/BM25_test_label.json')
    parser.add_argument('--num_process', type=int, default=10)
    args, _ = parser.parse_known_args()

    with open(args.train_file) as f:
        data = json.load(f)

    datas = split(data, args.num_process)
    assert sum(len(x) for x in datas) == len(data)

    es = Elasticsearch(hosts=args.host, port=args.port)

    jobs = []
    manager = Manager().dict()
    for i in range(args.num_process):
        job = Process(target=main, args=(datas[i], es, args.index, args.size, i, manager))
        jobs.append(job)
        job.start()

    for job in jobs:
        job.join()

    result = []
    for i in range(args.num_process):
        print(f'merge {i}th data')
        result = manager[i]
    print(len(result))
    result, labels = post_process(result)
    p1, p3 = PrecisionAtNum(1), PrecisionAtNum(3)
    p5, p10 = PrecisionAtNum(5), PrecisionAtNum(10)
    p20 = PrecisionAtNum(20)
    for label in labels:
        p1(label)
        p3(label)
        p5(label)
        p10(label)
        p20(label)
    print(p1)
    print(p3)
    print(p5)
    print(p10)
    print(p20)

    with open(args.out_path, 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
