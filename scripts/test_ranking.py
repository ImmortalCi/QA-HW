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

    count = 0

    for query, candidates in tqdm(test.items()):
        labels = [1 if c[0] == 'label ' else 0 for c in candidates[:30]]
        can_list = []
        for candidate in candidates:
            if type(candidate) == list:  # 列表说明有candidate
                if candidate[0] == 'label ':
                    can_list.append(candidate[1][0])
                else:
                    can_list.append(candidate[0])

        # candidates = [c[1][0] if c[0] == 'label ' else c[0] for c in candidates[:30]]
        # print(query)
        # print(labels[:10])
        # print(candidates[:10])
        # candi_ids = [s2id[c] for c in candidates]
        # cluster_size = [1] * len(candi_ids)
        all_candidates, cluster_size = [], []
        for c in can_list:
            all_candidates.extend(s2cluster[c])
            cluster_size.append(len(s2cluster[c]))
        candi_ids = [s2id[x] for x in all_candidates]
        candi_embs = embeddings[candi_ids]
        query_emb = model.encode([query])
        # candi_embs = model.encode(candidates)
        # label_embedding = model.encode([candidates[0]])
        # print(cosine_similarity(query_emb, label_embedding))
        # print('⬆️上面正常应该输出1')
        # print(label_embedding[0] == candi_embs[0])
        # print('⬆️上面正常应该输出True')
        # print(cosine_similarity(candi_embs[0:1], query_emb))
        # print(cosine_similarity(candi_embs[1:2], query_emb))
        # print(query_emb.sum())
        # print(candi_embs[1].sum(), sentences[candi_ids[1]])
        query_emb = query_emb.expand(len(candi_ids), -1)
        similarity = cosine_similarity(query_emb, candi_embs)
        # print(query)
        # print(similarity[:10])
        # print(candidates[:10])
        # print('```````````````````````````````````````````````')
        # count += 1
        # if count == 100:
        #     break
        similarity = similarity.split(cluster_size, -1)
        similarity = pad_sequence(similarity, True, -1)
        scores, indices = similarity.max(-1)
        scores, indices = scores.sort(dim=-1, descending=True)
        sorted_labels = [labels[i] for i in indices]
        # if sorted_labels != labels:
        #     print(query, sorted_labels.index(1))
        #     print(labels, sorted_labels)
        result.append(sorted_labels)
        # result.append(labels)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='faiss test'
    )
    parser.add_argument('--train_file', default='data/simCLUE_train.json',
                        help='train file')
    parser.add_argument('--test_file', default='/data1/wtc/HW-QA/fasttext_test_label.json',
                        help='test file')
    parser.add_argument('--device', default='7',
                        help='device')
    parser.add_argument('--save_path', default='save/stage2_fasttext_3.15',
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
    # # 取训练集前1000个评估
    # count = 0
    # new_test = dict()
    # for key, value in test_data.items():
    #     new_test[key] = value
    #     count += 1
    #     if count==1000:
    #         break
    # test_data = new_test
    labels = main(train_data, test_data, recaller)

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
