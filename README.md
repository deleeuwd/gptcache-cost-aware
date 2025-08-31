# GPTCache with a Smart Cost-Aware Eviction Policy

[![Release](https://img.shields.io/pypi/v/gptcache?label=Release&color&logo=Python)](https://pypi.org/project/gptcache/)
[![pip download](https://img.shields.io/pypi/dm/gptcache.svg?color=bright-green&logo=Pypi)](https://pypi.org/project/gptcache/)
[![Codecov](https://img.shields.io/codecov/c/github/zilliztech/GPTCache/dev?label=Codecov&logo=codecov&token=E30WxqBeJJ)](https://codecov.io/gh/zilliztech/GPTCache)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit/)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/zilliz_universe.svg?style=social&label=Follow%20%40Zilliz)](https://twitter.com/zilliz_universe)
[![Discord](https://img.shields.io/discord/1092648432495251507?label=Discord&logo=discord)](https://discord.gg/Q8C6WEjSWV)

---

This is an extension of **[GPTCache](https://github.com/zilliztech/GPTCache)**, created as part of an academic project. The primary goal of this project is to introduce and implement an enhanced Eviction Policy for caching environments in Large Language Models (LLMs).

**The Problem:** Traditional eviction policies like **Least Recently Used (LRU)** are not efficient enough for LLMs. They ignore the creation cost of each cached item and may evict computationally expensive items in favor of cheaper ones, simply because the cheaper item was accessed more recently.

**Our Solution:** We designed and implemented a suite of **Cost-Aware Eviction Policies** that integrate multiple signals beyond recency:

1. **Creation Cost (Latency):** expensive-to-generate responses should be preserved.
2. **Access Frequency (Popularity):** frequently used items should remain accessible.
3. **Adaptivity Over Time:** different decay and scaling functions were tested to avoid runaway effects.

Among the many strategies explored, one in particular emerged as especially effective:

**Log-Frequency Cost-Aware Cache:**

Items are scored by `score(k) = τ * cost(k) * log(1 + freq(k))`

balancing frequency and cost with logarithmic scaling.
This formulation was refined from an earlier recency-weighted version, and benchmarks showed that **recency-free LOGF consistently outperformed LRU and LFU in multiple workloads**.

## 🚀 Key Features of the Extension

* **Smart Eviction Policy (`CACache`):** Replaces the standard LRU with a mechanism that maximizes the economic value of the cache. Demonstrated significant improvements in *hit rate* and *throughput* on repetitive and long workloads, while maintaining comparable latency to baselines.
* **Full Integration:** Implemented directly inside `GPTCache `. built on `cachetools` and exposed as a drop-in eviction policy compatible with `GPTCache's` API.
* **Modularity and Flexibility:** `CACache` class can be customized for specific needs through the file `./gptcache/manager/eviction/cost_aware.py`.
* **Comprehensive Benchmark Suite:** We developed a test suite that allows for easy comparison between different eviction policies under various workloads.

## 😊 Quick Start with the New Policy

### dev install

```bash
# clone GPTCache repo
git clone -b dev https://github.com/zilliztech/GPTCache.git
cd GPTCache

# install the repo
pip install -r requirements.txt
python setup.py install
```

If you prefer to run the project using Docker, see [DEV-DOCKER.md](DEV-DOCKER.md) for instructions.

### Testing & Benchmark

You can run the unit tests for the cost-aware eviction implementation and execute the benchmark locally.

- To run the unit tests for `cost_cache.py`:

```bash
python -m pytest tests/unit_tests/eviction/test_cost_cache.py -q
```

- T run the eviction policy benchmark (basic):

```bash
cd benchmark
python eviction_benchmark.py
```

For more information about the benchmark options, flags and workloads, see [EVICTION_BENCH.md](benchmark/EVICTION_BENCH.md).

### example usage

To use the new eviction policy, you need to specify `eviction="CA"`(Cost Aware) when initializing the `DataManager`.

```python
from gptcache import cache
from gptcache.embedding import Onnx
from gptcache.manager import get_data_manager
from gptcache.adapter import openai

# General settings
onnx = Onnx()

# Initialize the DataManager with the new policy
# Note the eviction="CostAware" parameter
data_manager = get_data_manager(
    scalar_store="sqlite",
    vector_store="faiss",
    eviction="CA",
    vector_params={"dimension": onnx.dimension},
    eviction_params={"maxsize": 50}  # The cache size can be changed
)

# General GPTCache initialization
cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=data_manager
)
cache.set_openai_key()

# From here on, usage is completely standard
response = openai.ChatCompletion.create(
  model='gpt-3.5-turbo',
  messages=[
    {
        'role': 'user',
        'content': "What is Cost-Aware Eviction?"
    }
  ],
)
```
