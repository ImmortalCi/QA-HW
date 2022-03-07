class Metric:
    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __eq__(self, other):
        return self.score == other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    def __ne__(self, other):
        return self.score != other

    @property
    def score(self):
        raise AttributeError


class PrecisionAtNum(Metric):
    def __init__(self, num=1, eps=1e-5):
        super(PrecisionAtNum, self).__init__()
        self._num = num
        self._tp = 0
        self._total = 0.0
        self._eps = eps

    def __call__(self, labels):  # 累加计算p@n
        for i in range(self._num):
            if i >= len(labels):
                break
            if labels[i] == 1:
                self._tp = self._tp + 1
                break
        self._total = self._total + 1

    def __repr__(self):
        return f"r@{self._num} acc: {self.accuracy:.2%}\%"

    @property
    def score(self):
        return self.accuracy

    @property
    def tp(self):
        return self._tp

    @property
    def total(self):
        return self._total

    @property
    def accuracy(self):
        return self._tp/(self._total + self._eps)


