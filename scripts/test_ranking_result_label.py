"""
此脚本用于输出精排的评估结果, 输出的内容更翔尽一些，附带标签
"""
import argparse
import json
import os
import sys

sys.path.append('.')
import torch
from tqdm import tqdm
from ranker.recaller import Recaller
from ranker.utils.metric import PrecisionAtNum
from torch.nn.functional import cosine_similarity
from torch.nn.utils.rnn import pad_sequence


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


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
    case_dict = dict()

    for query, candidates in tqdm(test.items()):
        labels = [1 if c[0] == 'label ' else 0 for c in candidates]
        can_list = []
        for candidate in candidates:
            if type(candidate) == list:  # 列表说明有candidate
                if candidate[0] == 'label ':
                    can_list.append(candidate[1][0])
                else:
                    can_list.append(candidate[0])

        if len(can_list) == 0:  # 没有candidates的情况
            res = [0 for i in range(20)]
            result.append(res)
        else:
            all_candidates, cluster_size = [], []
            for c in can_list:
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
            result.append(sorted_labels)

            # 输出精排模型的评估结果
            lst = []
            for i, score in zip(indices, scores):
                if labels[i] == 1:
                    lst.append(['label', can_list[i], float(score)])
                else:
                    lst.append([can_list[i], float(score)])

            case_dict[query] = lst

    return result, case_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='case study'
    )
    parser.add_argument('--train_file', default='data/simCLUE_train.json',
                        help='train file')
    parser.add_argument('--test_file', default='/data1/wtc/HW-QA/fasttext_test_label.json',
                        help='test file')
    parser.add_argument('--device', default='6',
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

    labels, case_dict = main(train_data, test_data, recaller)

    with open('/data1/wtc/HW-QA/test_result_label.json', mode='w+', encoding='utf-8') as f:
        json.dump(case_dict, f, indent=4, ensure_ascii=False)

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
