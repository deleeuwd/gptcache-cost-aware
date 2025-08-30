# Cost-Aware Cache Eviction Policy

A smart eviction policy that prioritizes keeping expensive cache entries while removing cheaper ones. This implementation dynamically adjusts costs based on access patterns to adapt to usage over time.

## Quick Start

### Basic Usage

```python
from gptcache import cache, Config
from gptcache.manager import get_data_manager, CacheBase, VectorBase

# Use cost-aware eviction with default settings
data_manager = get_data_manager(
    CacheBase("sqlite"),
    VectorBase("faiss", dimension=384),
    max_size=100,
    eviction="CostAware"
)

cache.init(
    pre_func=get_prompt,
    data_manager=data_manager,
    embedding_func=embedding_func,
    similarity_evaluation=SearchDistanceEvaluation()
)
```

### Running the Benchmark

```bash
cd benchmark
# Quick test with cost-aware policy
python eviction_benchmark.py --policies CostAware --n 100 --max_size 15

# Compare all policies
python eviction_benchmark.py --policies LRU LFU FIFO RR CostAware --workloads repetitive novel
```

### Using a Custom Cost Function

```python
from gptcache.manager.eviction.cost_aware_cache import CostAwareCacheEviction

def my_custom_cost_func(base_cost, access_count, time_since_creation):
    """Custom cost function that heavily weights recent expensive operations."""
    hours_old = time_since_creation / 3600
    if hours_old < 2:  # Very recent items get big boost
        return base_cost * (2.0 ** access_count)
    else:
        return base_cost * (1.1 ** access_count)

# Create eviction policy with custom cost function
eviction = CostAwareCacheEviction(
    maxsize=100,
    cost_update_func=my_custom_cost_func
)

# Use it in data manager
data_manager = get_data_manager(
    CacheBase("sqlite"),
    VectorBase("faiss", dimension=384),
    max_size=100,
    eviction_base=eviction
)
```

## How It Works

### Core Concepts

1. **Base Cost**: Initial cost assigned when an entry is first cached
2. **Dynamic Cost**: Updated cost based on access patterns using a cost update function
3. **Eviction Strategy**: Always evicts the entry with the lowest current cost
4. **Access Tracking**: Monitors access frequency and timing to update costs

### Cost Update Algorithm

The default cost update function uses:

```python
def _default_cost_update_func(base_cost, access_count, time_since_creation):
    # Logarithmic frequency boost (diminishing returns)
    frequency_boost = 1 + math.log(1 + access_count) * 0.2
    
    # Age-based decay over 24 hours
    hours_old = time_since_creation / 3600
    if hours_old < 1:
        age_factor = 1.0  # Full boost for recent items
    else:
        age_factor = max(0.2, 1.0 - (hours_old - 1) / 23)
    
    # Apply multiplier (capped at 5x base cost)
    multiplier = min(frequency_boost * age_factor, 5.0)
    return base_cost * multiplier
```

### When Costs Are Updated

- **Cache Hit**: Every time an entry is accessed, its cost is recalculated
- **Cache Store**: When storing a new value for existing key, access count increases
- **Eviction Decision**: Costs are compared to find the lowest-cost entry to evict

## Implementation Details

### File Structure

```
gptcache/manager/eviction/
├── cost_aware_cache.py       # Main implementation
├── manager.py               # Eviction policy factory
└── base.py                  # Base eviction interface
```

### Key Classes

#### `CostAwareCache`
- Extends `cachetools.Cache` for robust cache management
- Tracks cost metadata for each entry
- Handles automatic eviction when cache is full

#### `CostAwareCacheEviction` 
- Implements `EvictionBase` interface for GPTCache integration
- Provides cost-aware eviction policy
- Supports custom cost update functions

### Integration Points

#### Data Manager Integration
In `gptcache/manager/data_manager.py`:
```python
# Cost information is passed to the eviction policy
if hasattr(self.eviction_base, 'policy') and self.eviction_base.policy == "CostAware" and costs is not None:
    self.eviction_base.put([(ids[i], costs[i]) for i in range(len(ids))])
else:
    self.eviction_base.put(ids)
```

#### Factory Integration  
In `gptcache/manager/factory.py`:
```python
if eviction == "CostAware" or eviction_base == "CostAware":
    eviction_base = EvictionBaseClass.get(
        name="CostAware",
        policy=eviction,
        maxsize=max_size,
        clean_size=clean_size if clean_size else int(max_size * 0.2)
    )
```

## Configuration Options

### Constructor Parameters

```python
CostAwareCacheEviction(
    maxsize=1000,                    # Maximum cache entries
    clean_size=200,                  # Legacy parameter (cachetools evicts one at a time)
    on_evict=None,                   # Callback when entries are evicted
    cost_update_func=None,           # Custom cost update function
    **kwargs                         # Additional arguments passed to cachetools
)
```

### Cost Update Function Signature

```python
def cost_update_func(base_cost: float, access_count: int, time_since_creation: float) -> float:
    """
    :param base_cost: Original cost when entry was first cached
    :param access_count: Total number of times this entry has been accessed  
    :param time_since_creation: Seconds elapsed since entry was created
    :return: New cost value
    """
    # Your custom logic here
    return updated_cost
```

## Customization Guide

### Changing the Default Cost Function

**File to edit**: `gptcache/manager/eviction/cost_aware_cache.py`

**Function to modify**: `_default_cost_update_func`

Example modifications:

```python
# More aggressive frequency boost
frequency_boost = 1 + math.log(1 + access_count) * 0.5  # Changed from 0.2

# Different age decay pattern  
age_factor = max(0.1, 1.0 / (1 + hours_old * 0.1))  # Exponential decay

# Higher cost multiplier cap
multiplier = min(frequency_boost * age_factor, 10.0)  # Changed from 5.0
```

### Creating Custom Cost Functions

#### Example 1: Token-based Costing
```python
def token_based_cost_func(base_cost, access_count, time_since_creation):
    """Cost function that treats base_cost as token count."""
    # Expensive operations (high token count) get bigger boost
    token_multiplier = 1 + (base_cost / 1000) * 0.1  # 10% boost per 1000 tokens
    access_boost = 1.2 ** min(access_count, 10)      # Cap access boost
    return base_cost * token_multiplier * access_boost
```

#### Example 2: Time-sensitive Costing  
```python
def time_sensitive_cost_func(base_cost, access_count, time_since_creation):
    """Heavily penalizes old entries to favor fresh content."""
    hours_old = time_since_creation / 3600
    if hours_old > 6:  # Entries older than 6 hours lose value fast
        return base_cost * 0.1 * (1.1 ** access_count)
    else:
        return base_cost * (1.5 ** access_count)
```

#### Example 3: Cost-based Prioritization
```python  
def cost_tier_func(base_cost, access_count, time_since_creation):
    """Different strategies based on cost tiers."""
    if base_cost > 50:      # Expensive operations
        return base_cost * (2.0 ** min(access_count, 5))  # Strong boost
    elif base_cost > 10:    # Medium cost operations  
        return base_cost * (1.5 ** min(access_count, 3))  # Moderate boost
    else:                   # Cheap operations
        return base_cost * (1.1 ** access_count)          # Minimal boost
```

### Using Custom Functions

#### Option 1: Direct Instantiation
```python
eviction = CostAwareCacheEviction(
    maxsize=100,
    cost_update_func=my_custom_cost_func
)

data_manager = get_data_manager(
    CacheBase("sqlite"),
    VectorBase("faiss", dimension=384), 
    eviction_base=eviction
)
```

#### Option 2: Factory with Custom Eviction
```python
# First create the custom eviction instance
custom_eviction = CostAwareCacheEviction(
    maxsize=100,
    cost_update_func=my_custom_cost_func
)

# Then pass it to the factory
data_manager = get_data_manager(
    cache_base="sqlite",
    vector_base="faiss",
    max_size=100,
    eviction_base=custom_eviction  # Use custom instance
)
```

## Testing and Validation

### Running Tests

```bash
# Test cost update functionality
python test_cost_update.py

# Benchmark cost-aware vs other policies
cd benchmark
python eviction_benchmark.py --policies CostAware LRU --workloads repetitive novel
```

### Monitoring Cost Updates

The eviction policy exposes internal state for debugging:

```python
# Check current costs and access patterns
for key, info in eviction._cache_info.items():
    print(f"{key}: cost={info['cost']:.2f}, "
          f"base_cost={info['base_cost']:.2f}, "
          f"access_count={info['access_count']}, "
          f"age={(time.time() - info['creation_time'])/3600:.1f}h")
```

### Debug Logging

Enable debug logging to see eviction decisions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

def debug_on_evict(keys):
    print(f"DEBUG: Evicted {len(keys)} entries: {keys}")

eviction = CostAwareCacheEviction(
    maxsize=10,
    on_evict=debug_on_evict
)
```

## Performance Characteristics

### Memory Overhead
- **Cost tracking**: ~200 bytes per cache entry
- **Access counters**: 8 bytes per entry  
- **Time tracking**: 8 bytes per entry

### Computational Overhead
- **Cost update**: O(1) per cache access
- **Eviction decision**: O(n) where n = cache size
- **Recommended for**: Caches where computation cost >> eviction cost

### When to Use Cost-Aware Eviction

**Good for**:
- LLM inference caching (high computation costs)
- Database query caching (expensive operations)
- API response caching (rate limits/cost per request)
- Image processing pipelines

**Not ideal for**:  
- Simple key-value storage (use LRU)
- Uniform cost operations (use LFU)
- Very high frequency access patterns (overhead may not be worth it)

## Advanced Configuration

### Eviction Callback Functions

```python
def advanced_eviction_callback(evicted_keys):
    """Log evicted entries with their cost information."""
    for key in evicted_keys:
        print(f"Evicted {key} - cost was important enough to track")
        # Could log to file, send metrics, etc.

eviction = CostAwareCacheEviction(
    maxsize=100,
    on_evict=advanced_eviction_callback
)
```

### Integration with External Cost Systems

```python
def external_cost_func(base_cost, access_count, time_since_creation):
    """Integrate with external cost tracking system."""
    # Get real-time cost from external system
    current_market_rate = get_current_llm_pricing()  # Your external function
    
    # Scale base cost by current market rates
    adjusted_base = base_cost * current_market_rate
    
    # Apply standard access pattern adjustments
    frequency_boost = 1 + math.log(1 + access_count) * 0.2
    hours_old = time_since_creation / 3600
    age_factor = max(0.2, 1.0 - hours_old / 24)
    
    return adjusted_base * frequency_boost * age_factor
```

## Migration Guide

### From Other Eviction Policies

```python
# Before (LRU)
data_manager = get_data_manager(
    cache_base="sqlite", 
    vector_base="faiss",
    max_size=100,
    eviction="LRU"
)

# After (Cost-Aware)  
data_manager = get_data_manager(
    cache_base="sqlite",
    vector_base="faiss", 
    max_size=100,
    eviction="CostAware"  # Just change this line!
)
```

### Providing Cost Information

When using the cache, provide cost information:

```python
# If your adapter supports cost information
cache_put(prompt, answer, cost=token_count * cost_per_token)

# Or track costs in your application layer
expensive_result = expensive_computation()
cache_put(key, (expensive_result, computation_cost))
```

## Troubleshooting

### Common Issues

1. **No cost information provided**: Entries default to cost 1.0
2. **Cost function returns NaN/inf**: Implement bounds checking in custom functions
3. **Memory usage growing**: Check that `on_evict` callback doesn't retain references

### Debug Tools

```python
# View current cache state
print("Cache size:", len(eviction._cache_info))
print("Memory usage:", sum(sys.getsizeof(v) for v in eviction._cache_info.values()))

# Check cost distribution
costs = [info['cost'] for info in eviction._cache_info.values()]
print(f"Cost range: {min(costs):.2f} - {max(costs):.2f}")
print(f"Average cost: {sum(costs)/len(costs):.2f}")
```

## Future Enhancements

### Possible Improvements

1. **Predictive Costing**: ML-based cost prediction
2. **Multi-dimensional Costs**: Memory, time, and financial costs
3. **Cost Budgets**: Eviction based on total cost thresholds
4. **Adaptive Parameters**: Self-tuning cost function parameters

### Contributing

To modify or extend the cost-aware implementation:

1. **Core logic**: `gptcache/manager/eviction/cost_aware_cache.py`
2. **Integration**: `gptcache/manager/factory.py` and `data_manager.py`
3. **Testing**: Add tests to `tests/unit_tests/manager/`
4. **Benchmarking**: Update `benchmark/eviction_benchmark.py`

---

*This implementation uses `cachetools.Cache` as the underlying cache data structure for reliability and efficiency.*