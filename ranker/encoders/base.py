import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()

        self._config = config
        self._pad_index = config.pad_index
        self._unk_index = config.unk_index

    def forward(self, text):
        raise NotImplementedError

    @classmethod
    def load(cls, fname, config):
        state = torch.load(fname, map_location='cpu')
        parser = cls(config)
        parser.load_state_dict(state)
        return parser

    def save(self, fname):
        torch.save(self.state_dict(), fname)

