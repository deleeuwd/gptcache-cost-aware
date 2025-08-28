"""
Cost-aware cache eviction implementation using cachetools.

This module provides a cost-aware eviction policy that prioritizes evicting 
cache entries with lower costs first. Costs are dynamically updated based on
access patterns (frequency and recency) to adapt to usage patterns.
"""

from typing import Any, Callable, List
import time
import math
import cachetools

from gptcache.manager.eviction.base import EvictionBase


def _default_cost_update_func(base_cost: float, access_count: int, time_since_creation: float) -> float:
    """
    Default cost update function that increases cost based on access frequency and recency.
    Uses bounded growth to prevent runaway cost inflation.
    
    :param base_cost: The original cost when the entry was first added
    :param access_count: Number of times this entry has been accessed
    :param time_since_creation: Time elapsed since entry was created (in seconds)
    :return: Updated cost value
    """
    # Use logarithmic scaling for frequency to prevent exponential growth
    # log(1 + access_count) provides diminishing returns for high access counts
    frequency_boost = 1 + math.log(1 + access_count) * 0.2  # Gentle bounded growth
    
    # Improved age decay: more gradual decay over 24 hours
    # Recent items (< 1 hour) get full boost, then gradual decay
    hours_old = time_since_creation / 3600
    if hours_old < 1:
        age_factor = 1.0  # Full boost for recent items
    else:
        # Gradual decay over 24 hours, minimum 20%
        age_factor = max(0.2, 1.0 - (hours_old - 1) / 23)
    
    # Calculate multiplier and cap it at 5x
    multiplier = min(frequency_boost * age_factor, 5.0)
    
    # Apply multiplier to base cost
    return base_cost * multiplier


class CostAwareCache(cachetools.Cache):
    """
    A cachetools-based cache that implements cost-aware eviction.
    
    This cache maintains cost information for each entry and evicts entries
    with the lowest cost first. Costs are dynamically updated based on
    access patterns using a configurable cost update function.
    
    When the cache reaches capacity, cachetools automatically calls popitem()
    to evict one entry, and we override this to select the lowest-cost entry.
    """
    
    def __init__(self, maxsize, cost_update_func=None, on_evict=None, getsizeof=None):
        super().__init__(maxsize, getsizeof)
        
        self._cost_update_func = cost_update_func or _default_cost_update_func
        self._on_evict = on_evict
            
        # Store metadata for each cache entry
        # Format: {cache_key: {'cost': float, 'base_cost': float, 'access_count': int, 'last_access': int, 'creation_time': float}}
        self._cache_info = {}
        self._access_counter = 0
        
    def __setitem__(self, key, value):
        """Set cache entry and track cost information."""
        # Extract cost from value if it's a (value, cost) tuple, otherwise default to 1.0
        if isinstance(value, tuple) and len(value) >= 2:
            actual_value, cost = value[0], value[1]
        else:
            actual_value, cost = value, 1.0
            
        self._access_counter = (self._access_counter + 1) % (2**63 - 1)  # Prevent overflow
        
        # Store the actual value in the cache (cachetools handles eviction automatically)
        super().__setitem__(key, actual_value)
        
        # Track cost information
        if key not in self._cache_info:
            # New entry
            current_time = time.time()
            self._cache_info[key] = {
                'cost': cost,
                'base_cost': cost,  # Store original cost for proper multiplier calculation
                'access_count': 1,
                'last_access': self._access_counter,
                'creation_time': current_time
            }
        else:
            # Update existing entry
            entry = self._cache_info[key]
            entry['access_count'] += 1
            entry['last_access'] = self._access_counter
            
            # Update cost using the cost update function with base_cost
            time_since_creation = time.time() - entry['creation_time']
            new_cost = self._cost_update_func(entry['base_cost'], entry['access_count'], time_since_creation)
            entry['cost'] = new_cost
            
    def __getitem__(self, key):
        """Get cache entry and update access information."""
        # Get the value using parent implementation
        value = super().__getitem__(key)
        
        # Update access information and cost
        if key in self._cache_info:
            self._access_counter = (self._access_counter + 1) % (2**63 - 1)  # Prevent overflow
            entry = self._cache_info[key]
            entry['access_count'] += 1
            entry['last_access'] = self._access_counter
            
            # Update cost based on access pattern using base_cost
            time_since_creation = time.time() - entry['creation_time']
            new_cost = self._cost_update_func(entry['base_cost'], entry['access_count'], time_since_creation)
            entry['cost'] = new_cost
            
        return value
    
    def __delitem__(self, key):
        """Delete cache entry and its cost information."""
        super().__delitem__(key)
        # Clean up cost information
        self._cache_info.pop(key, None)
        
    def popitem(self):
        """Remove and return a (key, value) pair using cost-aware eviction strategy."""
        if not self._cache_info:
            raise KeyError('popitem(): cache is empty')
            
        # Find the entry with lowest cost (deterministic tie-breaking)
        entries_with_priority = [
            (info['cost'], info['last_access'], info['creation_time'], cache_key)
            for cache_key, info in self._cache_info.items()
        ]
        
        # Get the entry with lowest cost (cost, last_access, creation_time for deterministic ordering)
        lowest_cost_entry = min(entries_with_priority, key=lambda x: (x[0], x[1], x[2]))
        key_to_evict = lowest_cost_entry[3]
        
        # Get the value and remove the entry
        value = super().__getitem__(key_to_evict)
        del self[key_to_evict]
        
        # Call eviction callback if provided
        if self._on_evict:
            self._on_evict([key_to_evict])
        
        return key_to_evict, value
    
    def clear(self):
        """Clear all entries from cache and cost information."""
        super().clear()
        self._cache_info.clear()
        self._access_counter = 0


class CostAwareCacheEviction(EvictionBase):
    """
    Cost-aware eviction policy that prioritizes evicting cache entries with lower costs.
    
    This eviction policy helps maximize the value of cached expensive operations while 
    removing cheaper ones when cache space is needed. Costs are dynamically updated 
    based on access patterns (frequency and recency) using a configurable cost update function.
    
    Built on top of cachetools.Cache for robust and efficient cache management.
    
    :param maxsize: Maximum number of entries in the cache
    :type maxsize: int
    :param clean_size: Legacy parameter for compatibility (cachetools evicts one item at a time)
    :type clean_size: int  
    :param on_evict: Callback function called when entries are evicted
    :type on_evict: Callable[[List[Any]], None]
    :param cost_update_func: Function to update costs based on access patterns
    :type cost_update_func: Callable[[float, int, float], float]
    """

    def __init__(
            self,
            maxsize: int = 1000,
            clean_size: int = 0,
            on_evict: Callable[[List[Any]], None] = None,
            cost_update_func: Callable[[float, int, float], float] = None,
            **kwargs,
    ):
        self._policy = "CostAware"
        self._maxsize = maxsize
        self._clean_size = clean_size if clean_size > 0 else int(maxsize * 0.2)
        
        # Create the underlying cachetools-based cache
        self._cache = CostAwareCache(
            maxsize=maxsize,
            cost_update_func=cost_update_func,
            on_evict=on_evict,
            **kwargs
        )
        
    @property
    def maxsize(self):
        """Get the maximum cache size."""
        return self._maxsize
        
    @property 
    def clean_size(self):
        """Get the clean size (legacy parameter for compatibility)."""
        return self._clean_size
        
    def put(self, objs: List[Any]):
        """
        Add cache entries to the eviction tracker.
        
        For cost-aware eviction, we expect objs to contain tuples of (cache_key, cost)
        where cost represents the computational/financial cost of generating the cached result.
        If cost is not provided, we default to 1.0.
        
        :param objs: List of cache keys or (cache_key, cost) tuples
        """
        for obj in objs:
            if isinstance(obj, tuple) and len(obj) >= 2:
                cache_key, cost = obj[0], obj[1]
                # Store as (value, cost) tuple - since we're just tracking existence, use True as value
                self._cache[cache_key] = (True, cost)
            else:
                cache_key = obj
                # Default cost, store as (value, cost) tuple
                self._cache[cache_key] = (True, 1.0)
                
    def get(self, obj: Any) -> bool:
        """
        Check if cache entry exists and update access information and cost.
        
        :param obj: Cache key or (cache_key, extra_data) tuple
        :return: True if entry exists, False otherwise
        """
        cache_key = obj[0] if isinstance(obj, tuple) else obj
        
        try:
            # Access the entry, which will trigger cost updates via __getitem__
            value = self._cache[cache_key]
            return True
        except KeyError:
            return False
            
    @property 
    def _cache_info(self):
        """Provide access to the underlying cache info for testing compatibility."""
        return self._cache._cache_info

    @property
    def policy(self) -> str:
        """Get the eviction policy name."""
        return self._policy
