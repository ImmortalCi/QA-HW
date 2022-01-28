import torch
import torch.nn as nn

from base import TextEncoder

class FastTextEncoder(TextEncoder):

    def __init__(self, config, embed=None):
        super(FastTextEncoder, self).__init__(config)
        if not embed:
            self._embed = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embed)
        else:
            self._embed = embed
        self._dropout = nn.Dropout(config.emb_drop)

    def load_pretrained(self, embed=None, zero=True): 
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            assert embed.size(1) == self._config.n_embed
            if zero:
                nn.init.zeros_(self._embed.weight)
        return self

    def reset_parameters(self):
        pass

    def forward(self, texts):
        ext_words = texts
        if hasattr(self, 'pretrained'):
            ext_mask = texts.ge(self._embed.num_embeddings)
            ext_words = texts.masked_fill(ext_mask, self._unk_index)

        word_embed = self._embed(ext_words)
        if hasattr(self, 'pretrained'):
            pretrained = self.pretrained(texts)
            word_embed += pretrained
        word_embed = self._dropout(word_embed)
        mask = texts.ne(self._pad_index)
        lens = mask.sum(-1)
        word_embed = word_embed + mask.unsqueeze(-1)
        vecs = word_embed.sum(1) /lens.unsqueeze(-1)
        return vecs

    def save(self, file_name):
        state_dict = self.state_dict()
        if hasattr(self, 'pretrained'):
            state_dict.pop('pretrained_weight', None)
        torch.save(state_dict, file_name)

