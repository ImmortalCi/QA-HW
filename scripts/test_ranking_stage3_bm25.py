"""
2021.4.7
因为ES-BM25和fast_text输出的格式不同，所以BM25下的评估脚本单独写了一下
这个脚本用于评估修正之后的评估结果
"""

import argparse
import json
import os
import sys

sys.path.append('.')
import faiss
import torch
from tqdm import tqdm
from ranker.recaller import Recaller
from ranker.utils.metric import PrecisionAtNum


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


def get_search_result(test_data, recaller, index, sentences, extend2standard, k):
    querys = [extend for _, extends in test_data.items() for extend in extends]
    golds = [standard for standard, extends in test_data.items()
             for _ in extends]
    vecs = recaller.encode(querys)
    vecs = vecs.cpu().numpy()
    faiss.normalize_L2(vecs)
    print('searching...')
    distance, item = index.search(vecs, k=125 * 30)
    result = [[sentences[j] for j in i[:30]] for i in item]
    labels = []
    assert len(result) == len(golds)
    for pred, gold in zip(result, golds):
        label = []
        for p in pred:
            if p == gold or gold in extend2standard.get(p, []):
                label.append(1)
            else:
                label.append(0)
        labels.append(label)

    labels_standard = []
    result = get_candidates(querys, item, sentences, extend2standard, 30)
    for pred, gold in zip(result, golds):
        label = []
        for p in pred:
            if p == gold or gold in extend2standard.get(p, []):
                label.append(1)
            else:
                label.append(0)
        labels_standard.append(label)
    return labels, labels_standard


def get_candidates(querys, item, sentences, extend2standard, k=30):
    all_candidates = []
    for i in range(len(querys)):
        candidates = []
        finished = False
        prev_standard = set()
        for candidate_idx in item[i]:
            candidate = sentences[candidate_idx]
            standards = extend2standard.get(candidate, [candidate])

            for standard in standards:
                if standard not in prev_standard:
                    prev_standard.add(standard)
                    candidates.append(candidate)

            if len(prev_standard) >= k:
                all_candidates.append(candidates)
                finished = True
                break
        if not finished:
            print(f'number {i} query only get {len(candidates)} candidates')
            all_candidates.append(candidates)
    return all_candidates


from torch.nn.functional import cosine_similarity
from torch.nn.utils.rnn import pad_sequence


def main(train, test, model):
    global count
    sentences = []
    s2cluster = {}
    for std, ext in train.items():
        cluster = []
        sentences.append(std)
        cluster.append(std)
        for e in ext:
            sentences.append(e)
            cluster.append(e)
        s2cluster[std] = cluster
        for e in ext:
            s2cluster[e] = cluster
    sentences = list(sorted(sentences))
    s2id = {s: i for i, s in enumerate(sentences)}
    # emb = []
    # for s in tqdm(sentences):
    #     emb.append(model.encode([s]))
    # embeddings = torch.cat(emb, dim=0)
    embeddings = model.encode(sentences)

    result = []
    case_dict1 = dict()
    case_dict2 = dict()

    cnt = 0

    test_data = test # 是一个list[dic]
    for dic in tqdm(test_data):
        for query, candidates in dic.items():
            labels = [1 if c[0] == 'labels' else 0 for c in candidates]
            candidates = [c[1] if c[0] == 'labels' else c for c in candidates]
            if len(candidates) == 0:  # 没有candidates的情况
                res = [0 for i in range(20)]
                result.append(res)
            else:
                all_candidates, cluster_size = [], []
                for c in candidates:
                    all_candidates.extend(s2cluster[c])
                    cluster_size.append(len(s2cluster[c]))
                candi_ids = [s2id[x] for x in all_candidates]
                candi_embs = embeddings[candi_ids]
                query_emb = model.encode([query])
                query_emb = query_emb.expand(len(candi_ids), -1)
                similarity = cosine_similarity(query_emb, candi_embs)
                similarity = similarity.split(cluster_size, -1)
                similarity = pad_sequence(similarity, True, -1)
                scores, indices = similarity.max(-1)
                scores, indices = scores.sort(dim=-1, descending=True)
                sorted_labels = [labels[i] for i in indices]
                try:  # 尝试去定位
                    position = sorted_labels.index(1)
                except ValueError:
                    position = 21
                # 可以修改区间来输出不同的数据
                if 0 < position <= 2:  # 定位在前三位但是不在第一位的query
                    # lst = [{'label_position': position}, []]
                    # for i in indices[:5]:   # 输出精排模型前五名的candidates
                    #     lst[1].append(can_list[i])
                    # case_dict[query] = lst
                    top1_emb = model.encode([candidates[indices[0]]])
                    label_emb = model.encode([candidates[indices[position]]])
                    score1 = cosine_similarity(top1_emb, label_emb)
                    score2 = cosine_similarity(query_emb[0], label_emb)
                    if score1 >= score2:
                        cnt += 1
                        print(f'第{cnt}个被修正的label')
                        sorted_labels[0] = 1
                        case_dict1[query] = [candidates[i] for i in indices[:3]]
                    else:
                        case_dict2[query] = [{"label_position": position}, [candidates[i] for i in indices[:3]]]
                result.append(sorted_labels)
    return result, case_dict1, case_dict2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='faiss test'
    )
    parser.add_argument('--train_file', default='data/simCLUE_train.json',
                        help='train file')
    parser.add_argument('--test_file', default='/data1/wtc/HW-QA/BM25_test_label.json',
                        help='test file')
    parser.add_argument('--device', default='6',
                        help='device')
    parser.add_argument('--save_path', default='save/stage2_bm25',
                        help='device')

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

    count = 0
    labels, case_dict1, case_dict2 = main(train_data, test_data, recaller)

    with open('/data1/wtc/HW-QA/BM25_case_study_amended.json', mode='w+', encoding='utf-8') as f1:
        json.dump(case_dict1, f1, indent=4, ensure_ascii=False)

    with open('/data1/wtc/HW-QA/BM25_case_study_unamended.json', mode='w+', encoding='utf-8') as f2:
        json.dump(case_dict2, f2, indent=4, ensure_ascii=False)

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

