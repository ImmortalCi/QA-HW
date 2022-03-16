import os
import random

import torch
from loguru import logger
from torch.nn import functional as F
from torch.nn.functional import cosine_similarity
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from tqdm import tqdm

from .encoders.BERT import BertEncoder
from .encoders.fast_text import FastTextEncoder
from .utils.config import Config
from .utils.corpus import Corpus, TrainingSamples, QA_Corpus, query_id2str, answer_str
from .utils.data import TextDataset, batchify
from .utils.metric import PrecisionAtNum
from .utils.tokenizer import CharTokenizer, PretrainedTokenizer
from .utils.vocab import Vocab

str2encoder = {"fast_text": FastTextEncoder, "bert": BertEncoder}


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


class Recaller(object):

    def __init__(self, config, vocab, encoder, optimizer=None, scheduler=None):
        super(Recaller, self).__init__()

        self._config = config
        self._vocab = vocab
        self._encoder = encoder
        self._optimizer = optimizer
        self._scheduler = scheduler
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
        if self._config.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
        torch.set_num_threads(self._config.thread)
        torch.manual_seed(self._config.seed)
        random.seed(self._config.seed)

        config_file = os.path.join(save_path, "config.ini")
        vocab_file = os.path.join(save_path, self._config.vocab_file)
        encoder_param = os.path.join(
            save_path, self._config.encoder_param_file)
        paths = {'config': config_file, 'vocab': vocab_file,
                 'encoder_param': encoder_param}
        if self._config.encoder == "bert":
            bert_config = os.path.join(save_path, self._config.bert_config)
            paths['bert_config'] = bert_config
        if split_eval:
            log_file = os.path.join(log_path, 'eval.log')
        else:
            log_file = os.path.join(log_path, 'train.log')
        logger.add(log_file, mode='w', level='INFO',
                   format='{time:YYYY/MM/DD HH:mm:ss}-{level}:{message}')

    def _prepare_data(self, train_file, split_eval):
        logger.info("read pair training from training file")
        if self._config.match_mode == 'query_query':
            if self._config.stage2_data == 'BM25':
                train_pair = Corpus.load_BM25(train_file)
            else:
                train_pair = Corpus.load(train_file)
        elif self._config.match_mode == 'query_answer':
            train_pair = QA_Corpus.load(train_file)
            queryid2str = query_id2str(self._config.question_file)
            answerid2str = answer_str(self._config.answer_file)
            logger.info("QA match mode, all corpus have been created")
        logger.info(f"There are total {len(train_pair)} here")

        if split_eval:
            logger.info("split train datasets for evaluation")
            train_pair, test_pair = train_pair.split(0.2)
            logger.info(f"Train: {len(train_pair)}" +
                        f"Test: {len(test_pair)}")
        else:
            train_pair, test_pair = train_pair, None

        logger.info("count tokens in the training file")
        if self._vocab is None:
            if self._config.encoder == 'bert':
                tokenizer = PretrainedTokenizer(
                    self._config.bert_path, self._config.remove_punctuation)
                vocab_file = os.path.join(self._config.bert_path, "vocab.txt")
                self._vocab = Vocab.from_file(vocab_file, tokenizer)
            else:
                tokenizer = CharTokenizer(self._config.remove_punctuation)  # 字符级分词器
                if self._config.match_mode == 'query_query':
                    self._vocab = Vocab.from_corpus(
                        tokenizer=tokenizer, corpus=train_pair, min_freq=1)
                elif self._config.match_mode == 'query_answer':
                    self._vocab = Vocab.QA_from_corpus(
                        tokenizer=tokenizer, corpus=train_pair, queryid2str=queryid2str, answerid2str=answerid2str)
        self._config.update(
            {'vocab_size': self._vocab.n_tokens, 'unk_index': self._vocab.unk_index})
        logger.info("Prepare triplets for training")
        if self._config.match_mode == 'query_query':
            if self._config.stage == 1:
                training_triplet = TrainingSamples.from_corpus(train_pair)
            elif self._config.stage == 2:
                if self._config.stage2_data == 'BM25':
                    training_triplet = TrainingSamples.from_corpus_stage2_BM25(train_pair)
                else:
                    training_triplet = TrainingSamples.from_corpus_stage2(train_pair)
        elif self._config.match_mode == 'query_answer':
            training_triplet = TrainingSamples.QA_from_corpus(train_pair, queryid2str, answerid2str)
        return training_triplet, train_pair, test_pair

    def _numericalize(self, triplets):
        trainset = TextDataset(*self._vocab.numericalize_triplets(triplets))

        train_loader = batchify(
            trainset, self._config.batch_size, shuffle=True, triplets=True)
        return train_loader

    def _create_models(self):

        if self._encoder is None:
            if self._config.encoder == "bert":
                bert_path = self._config.bert_path
                self._encoder = str2encoder[self._config.encoder](bert_path)
                self._encoder.load_pretrained(bert_path)
            else:
                self._encoder = str2encoder[self._config.encoder](self._config)

        self.to(self._config.device)

        self._optimizer = Adam(self._encoder.parameters(), self._config.lr)

    def _train_epoch(self, train_loader):

        loss = 0

        for querys, positives, negatives in tqdm(train_loader):
            self._optimizer.zero_grad()
            query_vecs = self._encoder(querys)
            positive_vecs = self._encoder(positives)
            negative_vecs = self._encoder(negatives)

            positive_scores = cosine_similarity(query_vecs, positive_vecs)
            negative_scores = cosine_similarity(query_vecs, negative_vecs)

            pair_loss = self._config.margin - positive_scores + negative_scores
            pair_loss = F.relu(pair_loss)

            pair_loss.mean().backward()
            loss += pair_loss.sum().item()
            clip_grad_norm_(self._encoder.parameters(), self._config.clip)
            self._optimizer.step()

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
        training_triplet, train_pair, test_pair = self._prepare_data(
            train_file, split_eval)
        # 数据转成tensor， 并组成batch
        train_loader = self._numericalize(training_triplet)
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

            loss = self._train_epoch(train_loader)
            change = abs(loss - prev_loss)

            if loss >= prev_loss or change < self._config.threshold:
                patience += 1
            else:
                patience = 0
                # 保存模型
                logger.info(f"epoch {epoch}: save the best model, loss={loss:.4f}")
                self.save(save_path)
                best_e = epoch
                prev_loss = float(loss)

            if patience >= self._config.patience:
                break

        logger.info(f"Best epoch: {best_e}, avg_loss:{prev_loss :.4f} ")
        self._init_status()

    def to(self, device):
        self._encoder = self._encoder.to(device)

    def save(self, save_path):
        if self._config.encoder == 'bert':
            self._encoder.save(save_path)
        else:
            encoder_path = os.path.join(
                save_path, self._config.encoder_param_file)
            self._encoder.save(encoder_path)

        config_path = os.path.join(save_path, 'config.ini')
        vocab_path = os.path.join(save_path, self._config.vocab_file)
        self._config.save(config_path)
        self._vocab.save(vocab_path)

    @classmethod
    def load(cls, path, device):

        config_path = os.path.join(path, 'config.ini')
        ranker_config = Config(config_path)
        vocab_path = os.path.join(path, ranker_config.vocab_file)
        encoder_param_path = os.path.join(
            path, ranker_config.encoder_param_file)

        if ranker_config.encoder == 'bert':
            tokenizer = PretrainedTokenizer(path)
        else:
            tokenizer = CharTokenizer(
                remove_punct=ranker_config.remove_punctuation)

        vocab = Vocab.load(vocab_path, tokenizer)

        if ranker_config.encoder == 'bert':
            encoder = str2encoder[ranker_config.encoder].load(path)
        else:
            encoder = str2encoder[ranker_config.encoder].load(
                encoder_param_path, ranker_config)

        encoder = encoder.to(device)

        encoder.eval()

        ranker = cls(config=ranker_config, vocab=vocab, encoder=encoder)
        return ranker

    @torch.no_grad()
    def encode(self, sents, batch_size=64):
        self._encoder.eval()

        device = next(self._encoder.parameters()).device
        vecs, batch = [], []
        for sent in sents:
            tokens = self._vocab.string2id(sent)
            batch.append(tokens)
            if len(batch) == batch_size:
                batch = pad_sequence(batch, True).to(device)
                vec = self._encoder(batch)
                vecs.append(vec)
                batch = []
        if len(batch) > 0:
            batch = pad_sequence(batch, True).to(device)
            vec = self._encoder(batch)
            vecs.append(vec)

        vecs = torch.cat(vecs, dim=0)
        return vecs