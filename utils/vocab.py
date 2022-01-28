from collections import Counter
from loguru import logger

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class Vocab(object):
    _unk = "[UNK]"
    _pad = "[PAD]"

    def __init__(self, tokens, tokenizer):
        self._tokens = tokens
        self._tokenizer = tokenizer
        self._token_dict = {token: i for i,token in enumerate(self._tokens)}

    @property
    def n_tokens(self):
        return len(self._tokens)

    def unk_index(self):
        return self._token_dict[self._unk]

    def __len__(self):
        return self.n_tokens

    def __repr__(self):
        s = f"{self.__class__.__name__}:"
        s += f"{self.n_tokens} tokens, "
        return s

    def string2id(self, string, lower=True, return_tensor=True):
        tokens = self._tokenizer.tokenize(string, lower)
        index = [self._token_dict.get(token, self.unk_index) for token in tokens]

        if return_tensor:
            return torch.tensor(index, dtype=torch.long)
        else:
            return index

    def numericalize_pairs(self, corpus):
        chars = [self.string2id(seq) for seq in corpus.users]
        candidates = []
        candidates = [pad_sequence([self.string2id(t[0]) for t in candidates]) for seq in corpus]
        labels = [torch.tensor([t[1] for t in seq.candidates], dtype=torch.long) for seq in corpus]
        return chars, candidates, labels

    def _numericalize_triplets(self, triplets, query, positive, negative, n):
        query_chars, positive_chars, negative_chars = [], [], []
        for triplet in triplets:
            query_chars.append(self.string2id(triplet.user_query, return_tensor=False))
            positive_chars.append(self.string2id(triplet.positive_query, return_tensor=False))
            negative_chars.append(self.string2id(triplet.negative_query, return_tensor=False))
        query[n] = query_chars
        positive[n] = positive_chars
        negative[n] = negative_chars

    def numericalize_triplets(self, triplets, n=5):
        all_sentences = set()
        for triplet in triplets:
            all_sentences.add(triplet.user_query)
            all_sentences.add(triplet.positive_query)
            all_sentences.add(triplet.negative_query)
        all_sentences = sorted(all_sentences)
        sent2id = {s: i for i, s in enumerate(all_sentences)}
        sent_ids = [self.string2id(s, True, True) for s in all_sentences]

        query_index, positive_index, negative_index = [], [], []
        for triplet in triplets:
            query_index.append(sent2id[triplet.user_query])
            positive_index.append(sent2id[triplet.positive_query])
            negative_index.append(sent2id[triplet.negative_query])

        return sent_ids, (query_index, positive_index, negative_index)

    @classmethod
    def from_corpus(cls, tokenizer, corpus, min_freq=1):
        special_tokens = [cls._pad, cls._unk]
        tokens = Counter(token.srip()
                         for seq in corpus.standards + corpus.extends for token in tokenizer.tokenize(seq)
                         if len(token.srip()) > 0 and token.strip() not in special_tokens) # 过滤掉只包含空格的tokens
        tokens = {token for token, freq in tokens.items() if freq>min_freq}
        tokens = special_tokens + sorted(tokens)
        vocab = cls(tokens, tokenizer)
        return vocab

    @classmethod
    def from_file(cls, filename, tokenizer):
        tokens = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tokens.append(line.strip())
        if tokens[0] != cls._pad:
            tokens.insert(0, cls._pad)
        if cls._unk not in set(tokens):
            tokens.append(cls._unk)
        vocab = cls(tokens, tokenizer)
        return vocab

    def load_pretrained_embedding(self):
        raise NotImplementedError

    def extend(self, tokens):
        self._tokens.extend(sorted(set(tokens).difference(set(self._tokens))))
        self._token_dict.update({token: i for i, token in enumerate(self._tokens)})

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for char in self._tokens:
                f.write(char)
                f.write('\n')

    @classmethod
    def load(cls, path, tokenizer):
        tokens = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens.append(line.strip())
        return cls(tokens, tokenizer)
