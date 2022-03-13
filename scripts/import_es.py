import argparse
import json

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
import jieba
import sys

sys.path.append(".")


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


def process(data):
    res = []
    for i, (standard, extends) in enumerate(data.items()):
        new_extends = []

        tokens = tokenize(standard)
        new_extends.append({
            'content': standard,
            'content_after_pre_tokenizer': ','.join(tokens),
        })

        for extend in extends:
            tokens = tokenize(extend)
            new_extends.append({
                'content': extend,
                'content_after_pre_tokenizer': ','.join(tokens)
            })
        res.append({
            'st_question': standard,
            'ex_question': new_extends
        })
    return res


def create_index(es, index):
    indexMappings = {
        "settings": {
            "index": {
                "analysis": {
                    "analyzer": {
                        "cn": {
                            "type": "pattern",
                            "lowercase": True,
                            "pattern": ","
                        }
                    },
                    "char_filter": {
                        "cn_pattern_punctuations_and_whitespace": {
                            "type": "pattern_replace",
                            "pattern": "\\pP|\\pS|\\s",
                            "replacement": ""
                        }
                    },
                    "normalizer": {
                        "cn_normalizer": {
                            "type": "custom",
                            "filter": [
                                "lowercase"
                            ],
                            "char_filter": [
                                "cn_pattern_punctuations_and_whitespace"
                            ]
                        }
                    }
                }
            }
        },
        "mappings": {
            "default_type": {
                "dynamic_templates": [
                    {
                        "new_fields": {
                            "match_mapping_type": "string",
                            "match": "*",
                            "unmatch": "relevance",
                            "mapping": {
                                "type": "keyword"
                            }
                        }
                    }
                ],
                "_source": {
                    "enabled": True
                },
                "properties": {
                    "st_question": {
                        "type": "keyword",
                    },
                    "ex_questions": {
                        "type": "nested",
                        "properties": {
                            "content_after_pre_tokenizer": {
                                "type": "text",
                                "similarity": "BM25",
                                "analyzer": "cn",
                                "fields": {
                                    "raw": {
                                        "type": "keyword",
                                    }
                                }
                            },
                            "content": {
                                "type": "keyword",
                                "fields": {
                                    "raw": {
                                        "type": "keyword",
                                        "normalizer": "cn_normalizer"
                                    },
                                    "raw_content": {
                                        "type": "keyword",
                                    }
                                }
                            }
                        }
                    },
                }
            }
        }
    }

    # 判断索引是否存在
    if not es.indices.exists(index=index):
        # 创建索引，制定索引名、索引·映射·
        es.indices.create(index=index, body=indexMappings)
        print("create index succeed")
    else:
        print("index has existed in the es")
        print("delete the old index")
        es.indices.delete(index)
        assert not es.indices.exists(index=index)
        es.indices.create(index=index, body=indexMappings)
        print("create index succeed")


def import_data(data, es, index):
    actions = []
    for i, item in enumerate(tqdm(data)):
        item["_index"] = index
        item['_id'] = i
        item["_type"] = "default_type"
        actions.append(item)
        if len(actions) == 1000:
            helpers.bulk(es, actions)
            actions = []
    if len(actions) > 0:
        helpers.bulk(es, actions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='import data to es'
    )
    parser.add_argument('--train_file', default='', help='train file')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=9200)
    parser.add_argument("--index", default='text-recall-es-hwnlp')
    args, _ = parser.parse_known_args()

    with open(args.train_file) as f:
        data = json.load(f)
    print('process data...')
    to_import_data = process(data)

    es = Elasticsearch(hosts=args.host, port=args.port)
    create_index(es, args.index)
    print("import data")
    import_data(to_import_data, es, args.index)
