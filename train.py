import argparse
import os
import torch

from recaller import Recaller

from utils.config import Config

def main(config):
    recaller = Recaller(config, None, None, None)
    recaller.train(config.train_file, config.save_path, config.save_path, split_eval=config.split_eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Text Recall")
    parser.add_argument('--encoder', help='encoder')
    parser.add_argument('--scorer', help='scorer')
    parser.add_argument('--device', default='', help='device')
    parser.add_argument('--remove_punctuation', action='store_true', help='remove_punctuation')
    parser.add_argument('--split_eval', action='store_true', help='split_eval')

    parser.add_argument('--n_embed', type=int, help='n_embed')
    parser.add_argument('--emb_drop', type=float, help='emb_drop')

    parser.add_argument('--n_filters', type=int)
    parser.add_argument('--n_layers', type=int)
    parser.add_argument('--n_lstm_hidden', type=int)
    parser.add_argument('--lstm_out_drop', type=float)
    parser.add_argument('--bert_path', help='bert path')

    parser.add_argument('--lr', type=float, help='lr')
    parser.add_argument('--clip', type=float, help='clip')
    parser.add_argument('--margin', type=float, help='margin')

    parser.add_argument('--seed', type=int)
    parser.add_argument('--thread', '-t', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--negative', type=int)

    parser.add_argument('--save_path', help='save path')
    parser.add_argument('--train_file', default='data/train.json')
    parser.add_argument('--test_file', default='data/test.json')
    parser.add_argument('--pretrained_embedding', default=None)
    parser.add_argument('--pretrained_vocab', default=None)

    args, _ = parser.parse_known_args()
    print(vars(args))

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
    main(config)
