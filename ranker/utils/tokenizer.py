import jieba
from transformers import AutoTokenizer

PUNCTUAION=frozenset()

def remove_punctuation(tokens):
    return [token for token in tokens if token not in PUNCTUAION]

class Tokenizer(object):
    def tokenize(self):
        raise NotImplementedError

class PretrainedTokenizer:
    def __init__(self, path, remove_punct=True):
        self._tokenizer = AutoTokenizer.from_pretrained(path, force_download=False)
        self._bos_token = self._tokenizer.cls_token
        self._eos_token = self._tokenizer.sep_token
        self._remove_punc = remove_punct

    def tokenize(self, string, lower=True, add_special_token=True):
        if lower:
            string = string.lower()
        tokens = self._tokenizer.tokenize(string)
        if self._remove_punc:
            tokens = remove_punctuation(tokens)
        if len(tokens) == 0:
            tokens = [string]
        if add_special_token:
            return [self._bos_token] + tokens + [self._eos_token]
        else:
            return tokens

class CharTokenizer:
    def __init__(self, remove_punct=True):
        self._bos_token = '[BOS]'
        self._eos_token = '[EOS]'
        self._remove_punc = remove_punct


    def tokenize(self, string, lower=True, add_special_token=False):
        if lower:
            string = string.lower()
        tokens = list(''.join(string.split()))
        if self._remove_punc:
            tokens = remove_punctuation(tokens)
        if len(tokens) == 0:
            tokens = [string]
        if add_special_token:
            return [self._bos_token] + tokens + [self._eos_token]
        else:
            return tokens

class jiebaTokenizer:
    def __init__(self, remove_punct=True):
        self._bos_token = '[BOS]'
        self._eos_token = '[EOS]'
        self._remove_punc = remove_punct

    def tokenize(self, string, lower=True, add_special_token=False):
        if lower:
            string = string.lower()
        tokens = list(jieba.cut(string))
        if self._remove_punc:
            tokens = remove_punctuation(tokens)
        if len(tokens) == 0:
            tokens = [string]
        if add_special_token:
            return [self._bos_token] + tokens + [self._eos_token]
        else:
            return tokens

    