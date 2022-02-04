import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def collate_fn(data):
    chars, candidates, labels = zip(*data)
    chars = pad_sequence(chars, True)
    candidates = pad_sequence(candidates, True).transpose(1, 2)

    if torch.cuda.is_available():
        chars = chars.cuda()
        candidates = candidates.cuda()

    return (chars, candidates, labels)


def triplet_collate_fn(data):
    reprs = (pad_sequence(i, True) for i in zip(*data))
    if torch.cuda.is_available():
        reprs = (i.cuda() for i in reprs)
    return reprs


class TextDataset(Dataset):
    def __init__(self, items, triplets):
        super(TextDataset, self).__init__()

        self._items = items
        self._triplets = triplets

    def __getitem__(self, index):
        return tuple(self._items[trip[index]] for trip in self._triplets)

    def __len__(self):
        return len(self._triplets[0])


def batchify(dataset, batch_size, shuffle=False, triplets=False):
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=triplet_collate_fn if triplets else collate_fn)
    return loader
