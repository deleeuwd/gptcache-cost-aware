import heapq
import math
from cachetools import Cache

class CACache(Cache):
    """
    Log-Frequency Cost-Aware Cache.
    Combines cost and frequency (no recency term):
      score = tau * cost * log(1 + freq)
    Evicts items with lowest score.
    """

    def __init__(self, maxsize, getsizeof=None, tau=75):
        super().__init__(maxsize, getsizeof)
        self._scores, self._costs = {}, {}
        self._freqs, self._gen = {}, {}
        self._heap = []
        self._tau = tau

    def __getitem__(self, key, cache_getitem=Cache.__getitem__):
        value = cache_getitem(self, key)
        self._touch(key)
        return value

    def __setitem__(self, key, cost, cache_setitem=Cache.__setitem__):
        cache_setitem(self, key, cost)
        if key in self._scores:
            self._costs[key] = float(cost)
            self._touch(key)
        else:
            self._insert(key, cost)

    def __delitem__(self, key, cache_delitem=Cache.__delitem__):
        cache_delitem(self, key)
        self._scores.pop(key, None)
        self._costs.pop(key, None)
        self._freqs.pop(key, None)
        self._gen.pop(key, None)

    def _touch(self, key):
        self._freqs[key] += 1
        cost = self._costs.get(key, 1.0)
        score = self._tau * cost * math.log1p(self._freqs[key])
        self._scores[key] = score
        self._gen[key] = self._gen.get(key, 0) + 1
        heapq.heappush(self._heap, (score, self._gen[key], key))

    def _insert(self, key, cost):
        self._scores[key] = self._tau * float(cost) * math.log1p(1)  # freq=1
        self._costs[key] = float(cost)
        self._freqs[key] = 1
        self._gen[key] = 1
        heapq.heappush(self._heap, (self._scores[key], 1, key))

    def popitem(self):
        while self._heap:
            score, gen, key = heapq.heappop(self._heap)
            if key in self._scores and self._gen.get(key) == gen:
                return (key, self.pop(key))
        raise KeyError(f"{type(self).__name__} is empty")
        