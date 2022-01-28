import os
import random

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from loguru import logger

from tqdm import tqdm

from utils.tokenizer import CharTokenizer, PretrainedTokenizer, jiebaTokenizer
from utils.metric import PrecisionAtNum
from utils.corpus import Corpus, TrainingSamples
from utils.vocab import Vocab
from utils.data import batchify, TextDataset
from utils.config import Config

from encoders.BERT import BertEncoder
from encoders.fast_text import FastTextEncoder

str2encoder = {{"fast_text": FastTextEncoder, "bert": BertEncoder}}


def cos_sim(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.sum(-1)
    q_loss = q_loss.sum(-1)

    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    p_loss = p_loss.mean()
    q_loss = q_loss.meac()

    loss = (p_loss + q_loss) / 2
    return loss


class Recaller(object):

    def __init__(self, config, vocab, encoder, scorer, optimizer=None, scheduler=None):
        super(Recaller, self).__init__()

        self._config = config
        self._vocab = vocab
        self._encoder = encoder
        self._scorer = scorer
        self._state = {}
        self._init_status()
        self._loss_func = BatchHardTripletLoss(margin=self._config.margin)

    def _init_status(self):
        self._state['training'] = False
        self._state['epoch'] = 0
        self._state['total_epoch'] = 0

    @property
    def config(self):
        return self._config

    def _initialize(self, save_path, log_path, split_eval):
        if self._config.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
        torch.set_num_threads(self._config.thread)
        torch.manual_seed(self._config.seed)
        random.seed(self._config.seed)

        config_file = os.path.join(save_path, "config.ini")
        vocab_file = os.path.join(save_path, self._config.vocab_file)
        encoder_param = os.path.join(save_path, self._config.encoder_param_file)
        scorer_param_path = os.path.join(save_path, self._config.scorer_param_file)
        paths = {'config': config_file, 'vocab': vocab_file, 'encoder_param': encoder_param,
                 'scorer_param': scorer_param_path}
        if self._config.encoder == "bert":
            bert_config = os.path.join(save_path, self._config.bert_config)
            paths['bert_config'] = bert_config
        if split_eval:
            log_file = os.path.join(log_path, 'eval.log')
        else:
            log_file = os.path.join(log_path, 'train.log')
        logger.add(log_file, mode='w', level='INFO', format='{time:YYYY/MM/DD HH:mm:ss}-{level}:{message}')

    def _prepare_data(self, train_file, split_eval, k):
        logger.info("read pair training from training file")
        train_pair = Corpus.load(train_file)
        logger.info(f"There are total {len(train_pair)} here")

        if split_eval:
            logger.info("split train datasets for evaluation")
            train_pair, test_pair = train_pair.split(0.2)
            logger.info(f"Train: {len(train_pair)}" + f"Test: {len(test_pair)}")
        else:
            train_pair, test_pair = train_pair, None

        logger.info("count tokens in the training file")
        if self._vocab is None:
            if self._config.encoder == 'bert':
                tokenizer = PretrainedTokenizer(self._config.bert_path, self._config.remove_punctuation)
                vocab_file = os.path.join(self._config.bert_path, "vocab.txt")
                self._vocab = Vocab.from_file(vocab_file, tokenizer)
            else:
                tokenizer = CharTokenizer(self._config.remove_punctuation)
                self._vocab = Vocab.from_corpus(tokenizer=tokenizer, corpus=train_pair, min_freq=1)
        self._config.update({'vocab_size': self._vocab.n_tokens, 'unk_index': self._vocab.unk_index})
        logger.info("Prepare triplets for training")
        training_triplet = TrainingSamples.from_corpus(train_pair, k)
        return training_triplet, train_pair, test_pair

    # def _numericalize(self, training_triplet, n):
    def _numericalize(self, training_pairs):
        logger.info("暂时不用多线程")

        # sent_ids, triplet = self._vocab.numericalize_triplets(training_triplet, n)
        # trainset = TextDataset(sent_ids, triplet)

        trainset = TextDataset(self._vocab.numericalize(training_pairs))

        train_loader = batchify(trainset, self._config.batch_size, shuffle=True, triplets=True)
        return train_loader

    def _create_models(self):

        if self._encoder is None:
            if self._config.encoder == "bert":
                bert_path = self._config.bert_path
                self._encoder = str2encoder[self._config.encoder](bert_path)
                self._encoder.load_pretrained(bert_path)
            else:
                self._encoder = str2encoder[self._config.encoder](self._config)
                if self._config.pretrained_embedding is not None and self._config.pretrained_vocab is not None:
                    embedding = self._vocab.load_pretrained_embedding(self._config.pretrained_vocab,
                                                                      self._config.pretrained_embedding)
                    self._encoder.load_pretrained(embedding)
            self._scorer = str2scorer[self._config.scorer](self._config)

        self.to(self._config.device)

        self._optimizer = Adam(self._encoder.parameters(), self._config.lr)

    def compute_nce_loss(self, pair_scores, cluster_sizes):
        pair_scores = pair_scores / self.config.temperature
        pair_scores = pair_scores.fill_diagonal_(-10000)
        n_clusters = len(cluster_sizes)
        if n_clusters == 1:
            return 0
        scores = pair_scores.new_zeros((n_clusters, n_clusters), dtype=torch.float)
        clusters = pair_scores.split(cluster_sizes, dim=0)
        for i, cluster_score in enumerate(clusters):
            cluster_pair_score = cluster_score.t().split(cluster_sizes, dim=0)
            cluster_pair_score = pad_sequence(cluster_pair_score, True, padding_value=-10000)
            sum_score = cluster_pair_score.max(dim=1)[0]
            sum_score = sum_score.sum(-1)
            scores[i] = sum_score
        label = torch.arange(n_clusters).to(scores.device)
        batch_loss = F.cross_entropy(scores, label)
        return batch_loss, scores

    def _train_epoch(self, train_loader):

        loss = 0

        for step, (querys, positives, negatives) in enumerate(train_loader):
            self._optimizer.zero_grad()
            query_vecs = self._encoder(querys)
            positive_vecs = self._encoder(positives)
            negative_vecs = self._encoder(negatives)

            positive_scores = self._scorer(query_vecs, positive_vecs)
            negative_scores = self._scorer(query_vecs, negative_vecs)

            pair_loss = self._config.margin - positive_scores + negative_scores
            pair_loss = F.relu(pair_loss)

            pair_loss.mean().backward()
            loss += pair_loss.sum().item()
            clip_grad_norm_(self._encoder.parameters(), self._config.clip)
            self._optimizer.step()

            if step % 1000 == 0:
                logger.info(f'epoch: {epoch} training step: {step}')
        loss /= len(train_loader.dataset)

        return loss
        # for clusters, labels in tqdm(train_loader):
        #     self._optimizer.zero_grad()
        #     vecs = self._encoder(clusters)
        #     batch_loss = self._loss_func(labels, vecs)
        #     batch_loss.backward()
        #     loss += batch_loss * len(clusters)
        #
        #     clip_grad_norm_(self._encoder.parameters(), self._config.clip)
        #
        #     self._optimizer.step()
        #
        # loss /=len(train_loader.dataset)
        # return loss

    def train(self, train_file, save_path, log_path, split_eval=False):
        self._init_status()
        self._state['training'] = True
        self._state['total_epoch'] = self._config.epochs

        # 初始化 gpu、种子、准备路径
        paths = self._initialize(save_path, log_path, split_eval)
        # 读取训练文件，统计辞典，用BM25生成三元组，根据split_eval参数决定是否切分
        training_triplet, train_pair, test_pair = self._prepare_data(train_file, split_eval, self._config.negative)
        # 数据转成tensor， 并组成batch
        train_loader = self._numericalize(training_triplet, n=1)
        # 创建初始模型
        self._create_models()
        # 保存初始模型
        self.save(save_path)

        prev_loss = float("inf")
        best_recall = PrecisionAtNum(20)
        patience, best_e = 0, 1

        # 开始训练
        for epoch in range(1, self._config.epochs + 1):
            self._state['epoch'] = epoch
            self._encoder.train()
            self._scorer.train()

            loss = self._train_epoch(train_loader)
            change = abs(loss - prev_loss)

            if loss >= prev_loss or change < self._config.threshold:
                patience += 1
            else:
                patience = 0
                # 保存模型
                logger.info("save the best model")
                self.save(save_path)
                best_e = epoch
                prev_loss = float(loss)

            if patience >= self._config.patience:
                break

        logger.info(f"Best epoch: {best_e}, avg_loss:{prev_loss :.4f} ")
        self._init_status()

    def to(self, device):
        self._encoder = self._encoder.to(device)
        self._scorer = self._scorer.to(device)

    def save(self, save_path):
        if self._config.encoder == 'bert':
            self._encoder.save(save_path)
        else:
            encoder_path = os.path.join(save_path, self._config.encoder_param_file)
            if hasattr(self._encoder, 'pretrained'):
                torch.save(self._encoder.pretrained.weight, os.path.join(save_path, self._config.pretrained_file))

        scorer_path = os.path.join(save_path, self._config.scorer_param_file)
        self._scorer.save(scorer_path)

        config_path = os.path.join(save_path, 'config.ini')
        vocab_path = os.path.join(save_path, self._config.vocab_file)
        self._config.save(config_path)
        self._vocab.save(vocab_path)

    @classmethod
    def load(cls, path, device):

        config_path = os.path.join(path, 'config.ini')
        ranker_config = Config(config_path)
        vocab_path = os.path.join(path, ranker_config.vocab_file)
        encoder_param_path = os.path.join(path, ranker_config.encoder_param_file)
        scorer_param_path = os.path.join(path, ranker_config.scorer_param_file)

        if ranker_config == 'bert':
            tokenizer = PretrainedTokenizer(path)
        else:
            tokenizer = CharTokenizer(remoce_punct=ranker_config.remove_punctuation)

        vocab = Vocab(vocab_path, tokenizer)

        if ranker_config.encoder == 'bert':
            encoder = str2encoder[ranker_config.encoder].load(path)
        else:
            encoder = str2encoder[ranker_config.encoder].load(encoder_param_path, ranker_config)
            if ranker_config.pretrained_embedding is not None and ranker_config.pretrained_vocab is not None:
                embedding = torch.load(os.path.join(path, ranker_config.pretrained_file), map_location='cpu')
                encoder.load_pretrained(embedding, False)
        scorer = str2scorer[ranker_config.scorer].load(scorer_param_path, ranker_config)

        encoder = encoder.to(device)
        scorer = scorer.to(device)

        encoder.eval()
        scorer.eval()

        ranker = cls(config=ranker_config, vocab=vocab, encoder=encoder, scorer=scorer)
        return ranker
