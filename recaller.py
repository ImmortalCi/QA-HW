import os

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from loguru import logger

from utils.tokenizer import CharTokenizer, PretrainedTokenizer, jiebaTokenizer
from utils.metric import PrecisionAtNum
from utils.corpus import Corpus, TrainingSamples
from utils.vocab import Vocab
from utils.data import batchify, TextDataset


class Recaller(object):

    def __init__(self, config, vocab, encoder, scorer, optimizer=None, scheduler=None):
        super(Recaller, self).__init__()

        self._config = config
        self._vocab = vocab
        self._encoder = encoder
        self._scorer = scorer
        self._state = {}
        self._init_status()

    def _init_status(self):
        self._state['training'] = False
        self._state['epoch'] = 0
        self._state['total_epoch'] = 0

    @property
    def config(self):
        return self._config

    def _initialize(self, save_path, log_path, split_eval):
        raise NotImplementedError

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

    def _numericalize(self, training_triplet, n):

        logger.info("暂时不用多线程")

        sent_ids, triplet = self._vocab.numericalize_triplets(training_triplet, n)
        trainset = TextDataset(sent_ids, triplet)

        train_loader = batchify(trainset, self._config.batch_size, shuffle=True, triplets=True)
        return train_loader

    def _create_models(self):

        if self._encoder is None:
            if self._config.encoder == "bert":
                bert_path = self._config.bert_path
                self._encoder =
                self._encoder.load_pretrained(bert_path)
            else:
                self._encoder = str2encoder[self._config.encoder](self._config)
                if self._config.pretrained_embedding is not None and self._config.pretrained_vocab is not None:
                    embedding = self._vocab.load_pretrained_embedding(self._config.pretrained_vocab, self._config.pretrained_embedding)
                    self._encoder.load_pretrained(embedding)
            self._scorer = str2scprer[self._config.scorer](self._config)

        self.to(self._config.device)

        self._optimizer = Adam(self._encoder.parameters(), self._config.lr)


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