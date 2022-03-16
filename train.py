import argparse
import os

import torch

from ranker.recaller import Recaller
from ranker.utils.config import Config


def main(config):
    recaller = Recaller(config, None, None, None)
    recaller.train(config.train_file, config.save_path, config.save_path, split_eval=config.split_eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Text Recall")
    parser.add_argument('--encoder', help='encoder')
    parser.add_argument('--device', default='5', help='device')
    parser.add_argument('--remove_punctuation', action='store_true', help='remove_punctuation')
    parser.add_argument('--split_eval', action='store_true', help='split_eval')

    parser.add_argument('--n_embed', type=int, help='n_embed')
    parser.add_argument('--emb_drop', type=float, help='emb_drop')
    parser.add_argument('--bert_path', help='bert path')

    parser.add_argument('--lr', type=float, help='lr')
    parser.add_argument('--clip', type=float, help='clip')
    parser.add_argument('--margin', type=float, help='margin')

    parser.add_argument('--seed', type=int)
    parser.add_argument('--thread', '-t', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--threshold', type=float)

    parser.add_argument('--save_path', help='save path')
    parser.add_argument('--train_file', default='data/train.json')
    parser.add_argument('--test_file', default='data/test.json')

    # query - query
    parser.add_argument('--match_mode', default='query_query')
    
    # query - answer
    # parser.add_argument('--match_mode', default='query_answer')
    # parser.add_argument('--question_file', default='data/question.csv')
    # parser.add_argument('--answer_file', default='data/answer.csv')

    # 精排模型
    parser.add_argument('--stage', default=2)  # 1代表初筛模型， 2代表精排模型

    # 精排模型训练集来源
    parser.add_argument('--stage2_data', default='1', type=str)

    args, _ = parser.parse_known_args()

    config = Config('default.ini')
    config.update(vars(args))

    if not os.path.exists(config.save_path):
        os.umask(0)
        os.makedirs(config.save_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if torch.cuda.is_available():
        config.update({'device': 'cuda'})
    else:
        config.update({'device': 'cpu'})
    torch.set_num_threads(config.thread)
    torch.manual_seed(config.seed)
    
    print('config:\n', config)
    main(config)
