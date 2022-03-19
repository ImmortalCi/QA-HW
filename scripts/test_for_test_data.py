import argparse
import json
import os

import sys
sys.path.append('.')
import faiss
import torch
from tqdm import tqdm
import numpy as np

from ranker.recaller import Recaller
from ranker.utils.metric import PrecisionAtNum

"""
用于fast_text初筛模型后的测试集生成，
"""

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def encode(sentences, recaller):
    vecs = []
    for sentence in sentences:
        tokens = recaller._vocab._tokenizer.tokenize(sentence)
        ids = torch.tensor([recaller._vocab._token_dict.get(token, recaller._vocab.unk_index)
                            for token in tokens])
        vec = recaller._encoder(ids.unsqueeze(0))
        vecs.append(vec)
    vecs = torch.cat(vecs, dim=0)
    return vecs


def import_faiss(data, recaller):
    vecs, sentences = [], []
    for standard, extends in data.items():
        sentences.append(standard)
        for extend in extends:
            sentences.append(extend)
    print('encoding...')
    vecs = recaller.encode(sentences)
    vecs = vecs.cpu().numpy()
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, sentences


def get_search_result(test_data, recaller, index, sentences, extend2standard, data_dict, k):
    querys = [extend for _, extends in test_data.items() for extend in extends]
    golds = [standard for standard, extends in test_data.items()
             for _ in extends]
    vecs = recaller.encode(querys)
    vecs = vecs.cpu().numpy()
    faiss.normalize_L2(vecs)
    print('searching...')
    distance, item = index.search(vecs, k=125*30)
    # result = [[sentences[j] for j in i[:30]] for i in item]
    # labels = []
    # assert len(result) == len(golds)
    # for pred, gold in tqdm(zip(result, golds)):
    #     label = []
    #     for p in pred:
    #         if p == gold or gold in extend2standard.get(p, []):
    #             label.append(1)
    #         else:
    #             label.append(0)
    #     labels.append(label)

    labels = None

    labels_standard = []
    result = get_candidates(querys, item, sentences, extend2standard, distance, golds, 20)
    print('Writing candidates...')
    for query, cand in tqdm(zip(querys, result)):
        data_dict[query] = cand
    for pred, gold in tqdm(zip(result, golds)):
        label = []
        for p in pred:
            if p[0] == gold or gold in extend2standard.get(p[0], []):
                label.append(1)
            else:
                label.append(0)
        labels_standard.append(label)
    for i, key in tqdm(enumerate(data_dict)):
        now_label = labels_standard[i]
        try:
            label_index = now_label.index(1)
            data_dict[key][label_index] = ('label ', data_dict[key][label_index])
        except ValueError:  # 前20个中没有正确答案
            data_dict[key].append('------前20个candidates中没有label------')
    return labels, labels_standard


def get_candidates(querys, item, sentences, extend2standard, distance, golds, k=30):
    all_candidates = []
    for i in tqdm(range(len(querys))):
        candidates = []
        finished = False
        prev_standard = set()
        for candidate_idx, score in zip(item[i], distance[i]):
            candidate = sentences[candidate_idx]
            standards = extend2standard.get(candidate, [candidate])
            for standard in standards:
                if standard not in prev_standard:
                    prev_standard.add(standard)
                    candidates.append((candidate, np.float(score)))

            if len(prev_standard) >= k:
                all_candidates.append(candidates)
                finished = True
                break

        if not finished:
            print(f'number {i} query only get {len(candidates)} candidates')
            all_candidates.append(candidates)
    return all_candidates

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='faiss test'
    )
    parser.add_argument('--train_file', default='data/simCLUE_train.json',
                        help='train file')
    parser.add_argument('--test_file', default='data/simCLUE_test.json',
                        help='test file')
    parser.add_argument('--device', default='6',
                        help='device')
    parser.add_argument('--save_path', default='save/stage1_3.15',
                        help='save path')

    args, _ = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    torch.set_num_threads(8)
    print('loading model...')
    recaller = Recaller.load(args.save_path, device=device)
    recaller.to(device)

    train_data = read_json(args.train_file)
    test_data = read_json(args.test_file)
    with torch.no_grad():
        index, sentences = import_faiss(train_data, recaller)
    max_extend = 125

    extend2standard = {}
    for standard, extends in train_data.items():
        for extend in extends:
            if extend not in extend2standard:
                extend2standard[extend] = [standard]
            else:
                print(
                    f'warning: {extend} appear in more than one standard questions')
                extend2standard[extend].append(standard)
    data_dict = dict()  # 用于保存每个query的candidates
    with torch.no_grad():
        labels, labels_standard = get_search_result(
            test_data, recaller, index, sentences, extend2standard, data_dict, k=30*max_extend)

    # p1, p3 = PrecisionAtNum(1), PrecisionAtNum(3)
    # p5, p10 = PrecisionAtNum(5), PrecisionAtNum(10)
    # p20, p30 = PrecisionAtNum(20), PrecisionAtNum(30)
    # for label in labels:
    #     p1(label)
    #     p3(label)
    #     p5(label)
    #     p10(label)
    #     p20(label)
    #     p30(label)
    # print(p1)
    # print(p3)
    # print(p5)
    # print(p10)
    # print(p20)
    # print(p30)
    # print()
    p1, p3 = PrecisionAtNum(1), PrecisionAtNum(3)
    p5, p10 = PrecisionAtNum(5), PrecisionAtNum(10)
    p20, p30 = PrecisionAtNum(20), PrecisionAtNum(30)
    for label in labels_standard:
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

    # for key, value in data_dict.items():
    #     print(key)
    #     print(value)
    #     exit()

    with open('/data1/wtc/HW-QA/fasttext_test_label.json', 'w+', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)

