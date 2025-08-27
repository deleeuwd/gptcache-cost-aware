from typing import Any, Callable, List
import heapq
from collections import defaultdict

from gptcache.manager.eviction.base import EvictionBase


class CostAwareCacheEviction(EvictionBase):
    """Cost-aware eviction: Evicts cache entries based on cost information
    
    This eviction policy prioritizes evicting cache entries with lower costs first,
    helping to maximize the value of cached expensive operations while removing
    cheaper ones when cache space is needed.

    :param maxsize: the maxsize of cache data
    :type maxsize: int
    :param clean_size: will clean the size of data when the size of cache data reaches the max size
    :type clean_size: int
    :param on_evict: the function for cleaning the data in the store
    :type  on_evict: Callable[[List[Any]], None]
    """

    def __init__(
            self,
            maxsize: int = 1000,
            clean_size: int = 0,
            on_evict: Callable[[List[Any]], None] = None,
            **kwargs,
    ):
        self._policy = "CostAware"
        self._maxsize = maxsize
        self._clean_size = clean_size if clean_size > 0 else int(maxsize * 0.2)
        self._on_evict = on_evict
        
        # Store cache entries with their costs and access info
        # Format: {cache_key: {'cost': float, 'access_count': int, 'last_access': int}}
        self._cache_info = {}
        # Min-heap to track entries by cost (cost, access_time, cache_key)
        self._cost_heap = []
        self._access_counter = 0
        
    @property
    def maxsize(self):
        """Get the maximum cache size"""
        return self._maxsize
        
    @property 
    def clean_size(self):
        """Get the clean size"""
        return self._clean_size
        
    def put(self, objs: List[Any]):
        """Add cache entries to the eviction tracker.
        
        For cost-aware eviction, we expect objs to contain tuples of (cache_key, cost)
        where cost represents the computational/financial cost of generating the cached result.
        If cost is not provided, we default to 1.0.
        """
        for obj in objs:
            if isinstance(obj, tuple) and len(obj) >= 2:
                cache_key, cost = obj[0], obj[1]
            else:
                cache_key, cost = obj, 1.0  # Default cost
                
            self._access_counter += 1
            
            if cache_key not in self._cache_info:
                # New entry
                self._cache_info[cache_key] = {
                    'cost': cost,
                    'access_count': 1,
                    'last_access': self._access_counter
                }
                heapq.heappush(self._cost_heap, (cost, self._access_counter, cache_key))
                
                # Check if we need to evict
                if len(self._cache_info) > self._maxsize:
                    self._evict_entries()
            else:
                # Update existing entry
                self._cache_info[cache_key]['access_count'] += 1
                self._cache_info[cache_key]['last_access'] = self._access_counter
                
        # Check if we need to evict entries
        if len(self._cache_info) > self._maxsize:
            self._evict_entries()

    def get(self, obj: Any) -> bool:
        """Check if cache entry exists and update access information."""
        cache_key = obj[0] if isinstance(obj, tuple) else obj
        
        if cache_key in self._cache_info:
            self._access_counter += 1
            self._cache_info[cache_key]['access_count'] += 1
            self._cache_info[cache_key]['last_access'] = self._access_counter
            return True
        return False

    def _evict_entries(self):
        """Evict entries based on cost-aware strategy."""
        entries_to_evict = []
        
        # Clean up the heap from already evicted entries
        while self._cost_heap and self._cost_heap[0][2] not in self._cache_info:
            heapq.heappop(self._cost_heap)
            
        # Collect entries to evict (prioritize lower cost items)
        temp_heap = []
        while len(entries_to_evict) < self._clean_size and self._cost_heap:
            cost, access_time, cache_key = heapq.heappop(self._cost_heap)
            
            if cache_key in self._cache_info:
                entries_to_evict.append(cache_key)
            # Don't add back to temp_heap as we're evicting these
                
        # Restore remaining entries to heap
        while temp_heap:
            heapq.heappush(self._cost_heap, temp_heap.pop())
            
        # Remove evicted entries from cache info
        for cache_key in entries_to_evict:
            if cache_key in self._cache_info:
                del self._cache_info[cache_key]
                
        # Call the eviction callback
        if self._on_evict and entries_to_evict:
            self._on_evict(entries_to_evict)

    @property
    def policy(self) -> str:
        return self._policy
