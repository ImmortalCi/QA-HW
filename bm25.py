import torch


class BM25(object):

    def __init__(self, docs):
        self._D = len(docs)
        self._dl = torch.tensor([len(doc) for doc in docs])
        self._avg_dl = self._dl.sum().float() / self._D
        self._words = set([word.lower() for doc in docs for word in doc])
        self._word2id = {word: i for i, word in enumerate(self._words)}
        self._n_words = len(self._words)

        f = []
        for doc in docs:
            word_ids = torch.tensor([self._word2id[word.lower()] for word in doc])
            count = (word_ids.unsqueeze(1) == torch.arange(self._n_words).unsqueeze(0)).sum(0)
            f.append(count)
        self._f = torch.stack(f).float()
        self._df = (self._f > 0).sum(dim=0)

        self._idf = torch.log((self._D - self._df + 0.5) / (self._df + 0.5))
        self._k1 = 1.2
        self._b = 0.75

        # to save memory
        del self._words
        del self._df

    def simall(self, doc):
        """

        :param doc(List[str]):
            分词后的query文档
        :return:
            generator(str):
                按分值从小到大生成文档
        """

        word_ids = torch.tensor([self._word2id.get(word.lower(), -1) for word in doc])
        mask = word_ids < 0
        fi = self._f[:, word_ids].transpose(0, 1)
        fi[mask] = 0

        k = self._k1 * (1 - self._b + self._b * self._dl / self._avg_dl)
        r = (fi * (self._k1 + 1)) /(fi + k)
        idf = self._idf[word_ids]

        scores = idf.unsqueeze(1) * r
        scores = scores.sum(0)
        scores, indices = torch.sort(scores, descending=True)
        return (i for i in indices)