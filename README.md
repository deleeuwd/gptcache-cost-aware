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

**Our Solution:** We developed a new eviction policy named `CostAwareCacheEviction`. This policy calculates a dynamic "value" for each item in the cache based on three key parameters:
1.  **Creation Cost (Base Cost):** How time-consuming and expensive it was to originally generate the response.
2.  **Access Frequency (Popularity):** How often this item is used.
3.  **Age:** How long the item has been in the cache, with its value decaying over time.

This way, we ensure that the most valuable and useful items remain in the cache, leading to significant performance improvements.

## 🚀 Key Features of the Extension

* **Smart Eviction Policy (`CostAwareCacheEviction`):** Replaces the standard LRU with a mechanism that maximizes the economic value of the cache.
* **Full Integration:** The new policy was built by inheriting from GPTCache's base classes, ensuring full compatibility and ease of use.
* **Modularity and Flexibility:** The cost calculation function can be customized for specific needs.
* **Comprehensive Benchmark Suite:** We developed a test suite that allows for easy comparison between different eviction policies under various workloads.

## 😊 Quick Start with the New Policy

To use the new eviction policy, you need to specify `eviction="CostAware"` when initializing the `DataManager`.

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
    eviction="CostAware",
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