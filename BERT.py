import torch.nn as nn

from transformers import BertConfig, BertModel

class BertEncoder(nn.Module):
    def __init__(self, path, pad_index=0):
        super(BertEncoder, self).__init__()
        bert_config = BertConfig.from_pretrained(path, force_download=False)
        self._bert = BertModel(bert_config)
        self._pad_index = pad_index

    def __repr__(self):  # 用于输出子模块信息
        n_layer = self._bert.config.num_hidden_layers
        n_hidden = self._bert.config.hidden_size
        pad_index = self._pad_index
        s = f"n_layers={n_layer}, n_hidden={n_hidden}, pad_index={pad_index}"
        return f"{self.__class__.__name__}({s})"

    def load_pretrained(self, bert_path):
        self._bert = BertModel.from_pretrained(bert_path, force_download=False)
        for param in self._bert.parameters():
            param.requires_grad = True

    def save(self, path):
        self._bert.save_pretrained(path)

    @classmethod
    def load(cls, path):
        encoder = cls(path)
        encoder.load_pretrained(path)
        return encoder

    def forward(self, ids):
        bert_mask = ids.ne(self._pad_index)
        output = self._bert(ids, attention_mask=bert_mask)
        last_hidden = output['last_hidden_state']
        lens = bert_mask.sum(1)
        last_hidden[~bert_mask] = last_hidden[~bert_mask] * 0.0
        bert_embed = last_hidden.sum(1)/lens.unsqueeze(-1)
        return bert_embed

