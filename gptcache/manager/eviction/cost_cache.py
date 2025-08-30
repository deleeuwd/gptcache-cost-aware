import heapq
from cachetools import Cache


class CACache(Cache):
    """ILP-inspired Cost-Aware cache: score = recency + accumulated cost."""

    def __init__(self, maxsize, getsizeof=None):
        Cache.__init__(self, maxsize, getsizeof)
        self._time = 0                # monotonic counter for recency
        self._scores = {}             # key -> score
        self._costs = {}              # key -> cost (LLM latency)
        self._heap = []               # min-heap of (score, key)
        self._offset = 0              # normalization offset (lazy)

    def __getitem__(self, key, cache_getitem=Cache.__getitem__):
        value = cache_getitem(self, key)
        self._touch(key)
        return value

    def __setitem__(self, key, value, cache_setitem=Cache.__setitem__):
        # value = cost (LLM latency)
        if key in self._scores:
            # already present: just update
            cache_setitem(self, key, value)
            self._touch(key)
            return

        if len(self) >= self.maxsize:
            self.popitem()

        cache_setitem(self, key, value)
        self._insert(key, value)

    def __delitem__(self, key, cache_delitem=Cache.__delitem__):
        cache_delitem(self, key)
        self._scores.pop(key, None)
        self._costs.pop(key, None)
        # heap entry left behind → handled lazily

    def _touch(self, key):
        """Update score on hit: add cost to score."""
        cost = self._costs.get(key, 1)
        self._scores[key] += cost
        heapq.heappush(self._heap, (self._scores[key], key))

    def _insert(self, key, cost):
        """Insert new item with score = current time"""
        self._time += 1
        score = self._time
        self._scores[key] = score
        self._costs[key] = cost
        heapq.heappush(self._heap, (score, key))

    def popitem(self):
        """Evict the item with the lowest score (lazy heap cleanup)."""
        while self._heap:
            score, key = heapq.heappop(self._heap)
            # Skip stale entries
            if key in self._scores and self._scores[key] == score:
                # Normalize all scores by subtracting this score
                self._offset += score
                # Adjust dictionary scores
                for k in self._scores:
                    self._scores[k] -= score
                return (key, self.pop(key))
        raise KeyError("%s is empty" % type(self).__name__)