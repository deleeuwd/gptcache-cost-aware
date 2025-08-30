# Cost-Aware (CA) Eviction Policy in GPTCache

## 1. Purpose

The Cost-Aware eviction policy is designed to preferentially retain cache entries that are:

- Expensive to regenerate (high LLM latency),
- Frequently reused,
- Recently inserted.

It extends traditional recency/frequency notions with an application-specific notion of “cost” = wall‑clock latency of the LLM call that produced the item.

---

## 2. End-to-End Data Flow (Where Cost Is Measured and How It Propagates)

1. Request enters via `gptcache/adapter/adapter.py`.
2. Cache lookup is performed (embedding / key derivation) through the data manager and eviction layer.
3. If HIT:
   - Value returned.
   - Eviction layer is notified implicitly via `__getitem__`, increasing the item’s score.
4. If MISS:
   - Adapter calls the underlying LLM provider.
   - Latency is measured: `llm_latency = time.time() - start_time`.
   - Data (response payload + metadata + `llm_latency`) is passed to `data_manager.put(...)`.
5. `data_manager` stores:
   - Payload (answer text / tokens) in object/value store.
   - Embeddings / vector index (if enabled).
   - Scalar metadata (including possibly cost for analytics).
   - Calls eviction layer to register the key with the measured cost.
6. Eviction subsystem (from `gptcache/manager/eviction/manager.py` via `EvictionBase.get`) creates a `MemoryCacheEviction` (for in-memory policy) which wraps the policy object (`CACache` when CA is chosen).
7. When capacity pressure occurs (`len(cache) >= maxsize`), eviction removes lowest‑priority items according to CA scoring, and orchestrates cleanup of associated stored objects.

---

## 3. Components Involved

| Layer                | File / Class                                                    | Role                                                              |
| -------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------- |
| Adapter              | `gptcache/adapter/adapter.py`                                 | Measures latency; passes cost downstream.                         |
| Data Manager         | `gptcache/manager/data_manager.py`                            | Central ingress for inserts; forwards cost to eviction.           |
| Eviction Factory     | `gptcache/manager/eviction/manager.py` (`EvictionBase.get`) | Chooses memory / redis / no-op eviction backend.                  |
| Memory Eviction Impl | `gptcache/manager/eviction/memory_cache.py`                   | Wraps an in‑process policy cache (CA / LRU / LFU / FIFO / etc.). |
| Cost-Aware Policy    | `gptcache/manager/eviction/cost_cache.py` (`CACache`)       | Maintains scoring and performs evictions.                         |

---

## 4. `CACache` Internal Model (`gptcache/manager/eviction/cost_cache.py`)

### Stored State

- `_time`: Monotonic insertion counter (recency baseline).
- `_scores: Dict[key, float]`: Current score per key.
- `_costs: Dict[key, float]`: Cost (latency) per key (as last recorded).
- `_heap: List[(score, key)]`: Min-heap of historical (score, key) snapshots (lazy invalidation).
- `_offset`: Accumulated normalization (bookkeeping; not externally used yet).

### Operations

| Operation              | Mechanics                                                                                                                                                                                             |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Insert (miss path)     | If full:`popitem()` to evict. Assign base score = `_time` (current code) though comment suggests "time + cost" (see below). Push to heap.                                                         |
| Hit (`__getitem__`)  | Calls `_touch(key)`: `score += cost` (the stored latency). Pushes updated snapshot to heap (lazy).                                                                                                |
| Update existing key    | Treated like hit: cost can be overwritten, then `_touch`.                                                                                                                                           |
| Eviction (`popitem`) | Pop heap until a live `(score, key)` matches `_scores[key]`. Normalize: subtract victim score from all remaining `_scores` (O(n)). Remove victim from underlying cache. Return `(key, cost)`. |

### Scoring Intuition

`score = recency_component + accumulated_cost_added_per_hit`

Each access adds the item’s original latency again. High-latency items gain protection faster with fewer hits; low-latency items must be accessed more often to survive.C

---

## 5. Eviction Semantics

- Victim = lowest current score.
- Retention preference hierarchy emerges naturally:
  1. Recently inserted items get higher base than older ones (monotonic `_time`).
  2. Frequently accessed items accumulate more cost increments.
  3. High-cost (slow) items get larger increments per hit than low-cost items.

Thus CA approximates a *recency + cost-weighted frequency* policy.

---

## 6. Separation of Payload vs. Eviction Index

`CACache` stores only the *cost* as the value. Actual response objects are stored elsewhere (data manager). On eviction:

- Eviction layer obtains the key (and cost value, if needed for metrics).
- Higher-level code must remove associated payload/metadata (embedding, answer text, vector index, scalar metadata) to reclaim memory/disk.

This decoupling keeps scoring structure lean but requires coordinated cleanup.

---

## 7. Complexity & Performance Notes

| Aspect         | Cost                                                                           |
| -------------- | ------------------------------------------------------------------------------ |
| Hit / Touch    | O(1) score update + O(log n) heap push (heap may expand due to staleness).     |
| Insert         | O(log n) + possible eviction cost.                                             |
| Evict (single) | O(log n + s + n) where `s` = stale heap pops; `n` from normalization loop. |
| Heap Growth    | Potentially multiple of live key count (one entry per access).                 |
| Memory         | Additional dicts + inflated heap from lazy invalidation.                       |

---



## 11. Example Code Adjustments

### Include Cost in Initial Score

```python
def _insert(self, key, cost):
	self._time += 1
	score = self._time + cost
	self._scores[key] = score
	self._costs[key] = cost
	heapq.heappush(self._heap, (score, key))
```

### Periodic Heap Rebuild Hook (Sketch)

```python
def _maybe_rebuild_heap(self):
	if len(self._heap) > 4 * len(self._scores):
		self._heap = [(s, k) for k, s in self._scores.items()]
		heapq.heapify(self._heap)

# Call _maybe_rebuild_heap() at end of _touch and _insert.
```

### Global Base Normalization (Conceptual)

Instead of subtracting victim score from every key (O(n)), maintain `self._base` and store per-key deltas; logical score = `self._base + delta`. On eviction set `self._base += evicted_score_increment` without rewriting all other entries. Requires adjusting comparisons using logical scores.

---

## 12. Summary

The CA eviction policy enhances GPTCache by leveraging runtime LLM latency as a proxy for regeneration cost. Its scoring—recency baseline plus cost-weighted accumulation per hit—tilts retention toward expensive, reused entries. Benchmarks show strongest gains in workloads where expensive results recur amidst cheaper noise (notably “novel-long” scenarios). Current implementation trade-offs (O(n) normalization, heap bloat, static cost) are acceptable for moderate cache sizes but can be optimized with lazy normalization and adaptive cost modeling.

---

---

For questions or iteration proposals, update this document with rationale and benchmark diffs.
