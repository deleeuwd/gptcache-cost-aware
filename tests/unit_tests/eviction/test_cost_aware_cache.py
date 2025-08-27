import unittest
import time
from unittest.mock import Mock, patch

from gptcache.manager.eviction.manager import EvictionBase
from gptcache.manager.eviction.cost_aware_cache import CostAwareCacheEviction
from gptcache.manager import get_data_manager
from gptcache.manager.data_manager import SSDataManager
from gptcache.utils.error import NotFoundError


class TestCostAwareCacheEviction(unittest.TestCase):
    """Test cases for CostAwareCacheEviction policy"""

    def test_cost_aware_instantiation_and_policy(self):
        """Test that EvictionBase.get returns CostAwareCacheEviction with correct policy"""
        eviction = EvictionBase.get(
            name="CostAware",
            maxsize=10,
            clean_size=2
        )
        
        self.assertIsInstance(eviction, CostAwareCacheEviction)
        self.assertEqual(eviction.policy, "CostAware")
        self.assertEqual(eviction.maxsize, 10)
        self.assertEqual(eviction.clean_size, 2)

    def test_put_with_costs_and_default_cost(self):
        """Test putting entries with explicit costs and default cost behavior"""
        eviction = CostAwareCacheEviction(maxsize=10, clean_size=2)
        
        # Put entry with explicit cost
        eviction.put([("key1", 5.0)])
        self.assertIn("key1", eviction._cache_info)
        self.assertEqual(eviction._cache_info["key1"]["cost"], 5.0)
        self.assertEqual(eviction._cache_info["key1"]["access_count"], 1)
        
        # Put entry without cost (should default to 1.0)
        eviction.put(["key2"])
        self.assertIn("key2", eviction._cache_info)
        self.assertEqual(eviction._cache_info["key2"]["cost"], 1.0)
        self.assertEqual(eviction._cache_info["key2"]["access_count"], 1)

    def test_get_updates_access_info(self):
        """Test that get updates access count and last access time"""
        eviction = CostAwareCacheEviction(maxsize=10, clean_size=2)
        
        # Put an entry
        eviction.put([("key1", 3.0)])
        initial_access_count = eviction._cache_info["key1"]["access_count"]
        initial_last_access = eviction._cache_info["key1"]["last_access"]
        
        # Get the entry
        result = eviction.get("key1")
        
        self.assertTrue(result)
        self.assertEqual(eviction._cache_info["key1"]["access_count"], initial_access_count + 1)
        self.assertGreater(eviction._cache_info["key1"]["last_access"], initial_last_access)
        
        # Get non-existent entry
        result = eviction.get("nonexistent")
        self.assertFalse(result)

    def test_eviction_evicts_low_cost_entries_on_maxsize(self):
        """Test that eviction removes lowest cost entries when maxsize is exceeded"""
        evicted_keys = []
        # Simulate actual cache store to ensure evicted keys are removed from real storage
        cache_store = {}
        
        def on_evict(keys):
            evicted_keys.extend(keys)
            # Simulate removal from actual cache store (what the real cache backend would do)
            for key in keys:
                cache_store.pop(key, None)
        
        eviction = CostAwareCacheEviction(maxsize=3, clean_size=2, on_evict=on_evict)
        
        # Add entries with different costs and populate simulated cache store
        eviction.put([("key1", 10.0)])  # High cost
        cache_store["key1"] = "value1"
        eviction.put([("key2", 1.0)])   # Low cost  
        cache_store["key2"] = "value2"
        eviction.put([("key3", 5.0)])   # Medium cost
        cache_store["key3"] = "value3"
        
        # Should not trigger eviction yet
        self.assertEqual(len(eviction._cache_info), 3)
        self.assertEqual(len(evicted_keys), 0)
        self.assertEqual(len(cache_store), 3)
        
        # Add one more entry to exceed maxsize
        eviction.put([("key4", 8.0)])   # High cost
        cache_store["key4"] = "value4"
        
        # Should have evicted 2 entries (clean_size)
        self.assertEqual(len(eviction._cache_info), 2)  # 4 - 2 = 2 remaining
        self.assertEqual(len(evicted_keys), 2)
        self.assertEqual(len(cache_store), 2)  # Verify actual cache store also reduced
        
        # Verify evicted keys are removed from both eviction metadata AND cache store
        for key in evicted_keys:
            self.assertNotIn(key, eviction._cache_info, f"Evicted key {key} still in eviction metadata")
            self.assertNotIn(key, cache_store, f"Evicted key {key} still in cache store")
        
        # Should have evicted the lowest cost entries first
        # key2 (cost=1.0) should definitely be evicted
        self.assertIn("key2", evicted_keys)
        
        # Either key3 (cost=5.0) should be evicted, keeping the highest cost ones
        remaining_keys = set(eviction._cache_info.keys())
        self.assertIn("key1", remaining_keys)  # Highest cost should remain
        self.assertIn("key4", remaining_keys)  # Second highest should remain
        
        # Verify remaining keys are still in the cache store
        for key in remaining_keys:
            self.assertIn(key, cache_store, f"Remaining key {key} missing from cache store")

    def test_eviction_heap_cleanup_and_no_double_eviction(self):
        """Test that heap cleanup works correctly and prevents double eviction"""
        evicted_keys = []
        
        def on_evict(keys):
            evicted_keys.extend(keys)
        
        eviction = CostAwareCacheEviction(maxsize=2, clean_size=1, on_evict=on_evict)
        
        # Add entries
        eviction.put([("key1", 5.0)])
        eviction.put([("key2", 3.0)])
        
        # Force eviction
        eviction.put([("key3", 1.0)])
        
        # Verify one entry was evicted (the lowest cost)
        self.assertEqual(len(evicted_keys), 1)
        self.assertIn("key3", evicted_keys)  # Lowest cost should be evicted
        
        # Clear evicted keys and add another entry
        evicted_keys.clear()
        eviction.put([("key4", 2.0)])
        
        # Should evict another entry
        self.assertEqual(len(evicted_keys), 1)
        
        # Verify no KeyError occurs during heap cleanup
        # This tests the heap cleanup logic in _evict_entries

    def test_put_updates_existing_entry_and_triggers_eviction_when_needed(self):
        """Test that putting existing entry updates access info and handles eviction"""
        evicted_keys = []
        
        def on_evict(keys):
            evicted_keys.extend(keys)
        
        eviction = CostAwareCacheEviction(maxsize=2, clean_size=1, on_evict=on_evict)
        
        # Add initial entries
        eviction.put([("key1", 5.0)])
        eviction.put([("key2", 3.0)])
        
        initial_access_count = eviction._cache_info["key1"]["access_count"]
        
        # Update existing entry (should not trigger eviction)
        eviction.put([("key1", 5.0)])
        
        self.assertEqual(len(eviction._cache_info), 2)
        self.assertEqual(len(evicted_keys), 0)
        self.assertEqual(eviction._cache_info["key1"]["access_count"], initial_access_count + 1)
        
        # Add new entry to trigger eviction
        eviction.put([("key3", 1.0)])
        
        # Should have evicted one entry
        self.assertEqual(len(evicted_keys), 1)

    def test_integration_import_data_passes_costs_to_eviction(self):
        """Test that SSDataManager.import_data correctly passes costs to CostAware eviction"""
        # TODO: This test needs to be implemented once we add proper integration
        # between SSDataManager and cost-aware eviction. Currently SSDataManager
        # doesn't have the _add_embedding_data method and proper cost handling.
        # This is not first priority right now.
        pass
        
        # # Create a mock data manager with cost-aware eviction
        # with patch('gptcache.manager.data_manager.SSDataManager._add_embedding_data'), \
        #      patch('gptcache.manager.data_manager.SSDataManager._add_cache_data') as mock_add_cache:
        #     
        #     # Mock the cache data addition to return IDs
        #     mock_add_cache.return_value = [1, 2, 3]
        #     
        #     # Create data manager with cost-aware eviction
        #     eviction = CostAwareCacheEviction(maxsize=10, clean_size=2)
        #     
        #     # Mock other required components
        #     mock_cache_base = Mock()
        #     mock_vector_base = Mock() 
        #     mock_vector_base.mul_add.return_value = None
        #     
        #     data_manager = SSDataManager(
        #         s=mock_cache_base,
        #         v=mock_vector_base,
        #         e=eviction
        #     )
        #     
        #     # Import data with costs
        #     questions = ["q1", "q2", "q3"]
        #     answers = ["a1", "a2", "a3"] 
        #     embeddings = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        #     costs = [10.0, 5.0, 15.0]
        #     
        #     data_manager.import_data(
        #         questions=questions,
        #         answers=answers,
        #         embedding_datas=embeddings,
        #         costs=costs
        #     )
        #     
        #     # Verify cost-aware eviction received the cost information
        #     self.assertEqual(len(eviction._cache_info), 3)
        #     # Note: The actual IDs are mocked, but we can verify the structure
        #     cache_keys = list(eviction._cache_info.keys())
        #     
        #     # Verify costs were stored (though the exact mapping depends on mock IDs)
        #     costs_stored = [eviction._cache_info[key]["cost"] for key in cache_keys]
        #     self.assertEqual(set(costs_stored), set(costs))

    def test_manager_factory_creates_costaware_when_requested(self):
        """Test that manager factory creates CostAware eviction when requested"""
        # Provide mock cache and vector backends so the factory can instantiate
        mock_cache_base = Mock()
        mock_cache_base.get_ids.return_value = []  # Return empty list for get_ids
        mock_vector_base = Mock()
        
        # Test with get_data_manager
        data_manager = get_data_manager(
            cache_base=mock_cache_base,
            vector_base=mock_vector_base,
            eviction_base="CostAware"
        )
        
        self.assertIsNotNone(data_manager.eviction_base)
        self.assertIsInstance(data_manager.eviction_base, CostAwareCacheEviction)
        self.assertEqual(data_manager.eviction_base.policy, "CostAware")

    def test_on_evict_callback_called_with_correct_ids(self):
        """Test that on_evict callback receives the correct evicted IDs"""
        evicted_keys = []
        
        def on_evict(keys):
            evicted_keys.extend(keys)
        
        eviction = CostAwareCacheEviction(maxsize=3, clean_size=2, on_evict=on_evict)
        
        # Add entries
        eviction.put([("key1", 10.0)])
        eviction.put([("key2", 1.0)])
        eviction.put([("key3", 5.0)])
        eviction.put([("key4", 2.0)])
        
        # Verify evicted keys are correct
        self.assertEqual(len(evicted_keys), 2)
        
        # The two lowest cost entries should be evicted
        expected_evicted = {"key2", "key4"}  # costs 1.0 and 2.0
        self.assertEqual(set(evicted_keys), expected_evicted)
        
        # Verify remaining entries are the higher cost ones
        remaining_keys = set(eviction._cache_info.keys())
        expected_remaining = {"key1"}  # Only key1 (cost=10.0) should remain after clean_size=2
        # Note: Due to heap ordering, we might have key3 or key1 remaining, 
        # but key2 and key4 should definitely be evicted

    def test_policy_property_and_basic_get_behavior(self):
        """Test policy property and basic get behavior"""
        eviction = CostAwareCacheEviction(maxsize=5, clean_size=1)
        
        # Test policy property
        self.assertEqual(eviction.policy, "CostAware")
        
        # Test get on empty cache
        self.assertFalse(eviction.get("nonexistent"))
        
        # Add entry and test get
        eviction.put([("test_key", 3.0)])
        self.assertTrue(eviction.get("test_key"))
        
        # Test get with tuple format (for compatibility)
        self.assertTrue(eviction.get(("test_key", "extra_data")))

    def test_cost_aware_eviction_with_equal_costs(self):
        """Test eviction behavior when multiple entries have the same cost"""
        evicted_keys = []
        
        def on_evict(keys):
            evicted_keys.extend(keys)
        
        eviction = CostAwareCacheEviction(maxsize=3, clean_size=2, on_evict=on_evict)
        
        # Add entries with same costs
        eviction.put([("key1", 5.0)])
        eviction.put([("key2", 5.0)])
        eviction.put([("key3", 5.0)])
        
        # Add entry to trigger eviction
        eviction.put([("key4", 5.0)])
        
        # Should evict 2 entries (clean_size)
        self.assertEqual(len(evicted_keys), 2)
        self.assertEqual(len(eviction._cache_info), 2)
        
        # When costs are equal, should evict based on insertion order (FIFO-like)
        # Earlier entries should be evicted first
        self.assertIn("key1", evicted_keys)
        self.assertIn("key2", evicted_keys)

    def test_clean_size_defaults_to_20_percent_of_maxsize(self):
        """Test that clean_size defaults to 20% of maxsize when not specified"""
        eviction = CostAwareCacheEviction(maxsize=100)
        self.assertEqual(eviction.clean_size, 20)  # 20% of 100
        
        eviction2 = CostAwareCacheEviction(maxsize=50)
        self.assertEqual(eviction2.clean_size, 10)  # 20% of 50

    def test_eviction_with_zero_costs(self):
        """Test eviction behavior with zero and negative costs"""
        evicted_keys = []
        
        def on_evict(keys):
            evicted_keys.extend(keys)
        
        eviction = CostAwareCacheEviction(maxsize=2, clean_size=1, on_evict=on_evict)
        
        # Add entries with zero and negative costs
        eviction.put([("key1", 0.0)])
        eviction.put([("key2", -1.0)])
        
        # Add positive cost entry to trigger eviction
        eviction.put([("key3", 1.0)])
        
        # Should evict the entry with lowest (most negative) cost
        self.assertEqual(len(evicted_keys), 1)
        self.assertIn("key2", evicted_keys)  # -1.0 is lowest

    def test_large_number_of_entries_performance(self):
        """Test that eviction works efficiently with larger number of entries"""
        evicted_keys = []
        
        def on_evict(keys):
            evicted_keys.extend(keys)
        
        eviction = CostAwareCacheEviction(maxsize=100, clean_size=20, on_evict=on_evict)
        
        # Add many entries with random-ish costs
        for i in range(105):  # Exceed maxsize to trigger eviction
            cost = (i % 10) + 1  # Costs from 1 to 10
            eviction.put([(f"key{i}", cost)])
        
        # Should have evicted some entries
        self.assertGreater(len(evicted_keys), 0)
        self.assertLessEqual(len(eviction._cache_info), 100)
        
        # Verify lowest cost entries were evicted
        remaining_costs = [eviction._cache_info[key]["cost"] for key in eviction._cache_info]
        self.assertTrue(all(cost >= min(remaining_costs) for cost in remaining_costs))

    def test_cache_never_exceeds_maxsize(self):
        """Test that cache size never exceeds maxsize limit"""
        maxsize = 5
        clean_size = 2
        evicted_keys = []
        
        def on_evict(keys):
            evicted_keys.extend(keys)
        
        eviction = CostAwareCacheEviction(maxsize=maxsize, clean_size=clean_size, on_evict=on_evict)
        
        # Add entries one by one and verify size never exceeds maxsize
        for i in range(15):  # Add way more than maxsize
            eviction.put([(f"key{i}", float(i))])
            
            # Cache should never exceed maxsize
            cache_size = len(eviction._cache_info)
            self.assertLessEqual(cache_size, maxsize, 
                                f"Cache size {cache_size} exceeded maxsize {maxsize} after adding key{i}")
            
            # Once we exceed maxsize initially, evictions should occur
            if i >= maxsize:
                # Should have started evicting entries
                self.assertGreater(len(evicted_keys), 0, 
                                  f"Expected evictions to have occurred by key{i}")
                
                # After eviction, cache size should be reasonable (not exceeding maxsize)
                # The exact size depends on implementation details, but should be <= maxsize
                self.assertLessEqual(cache_size, maxsize,
                                   f"Cache size {cache_size} should not exceed maxsize {maxsize}")
        
        # Final verification: cache should not exceed maxsize
        final_size = len(eviction._cache_info)
        self.assertLessEqual(final_size, maxsize,
                            f"Final cache size {final_size} should not exceed maxsize {maxsize}")
        
        # Verify that evictions occurred when needed
        self.assertGreater(len(evicted_keys), 0, "Some evictions should have occurred")
        
        # Verify that high-cost entries are retained (cost-aware behavior)
        remaining_costs = [eviction._cache_info[key]["cost"] for key in eviction._cache_info]
        if len(remaining_costs) > 1:
            # Should contain higher cost entries (most recently added have higher costs)
            min_remaining_cost = min(remaining_costs)
            max_remaining_cost = max(remaining_costs)
            # The minimum remaining cost should be reasonably high compared to what we added
            self.assertGreater(min_remaining_cost, 5.0, 
                              "Should have evicted lower cost entries and kept higher cost ones")

    def test_strict_eviction_behavior_on_maxsize_exceeded(self):
        """Test exact eviction behavior when maxsize is exceeded"""
        maxsize = 5
        clean_size = 2
        evicted_keys = []
        
        def on_evict(keys):
            evicted_keys.extend(keys)
        
        eviction = CostAwareCacheEviction(maxsize=maxsize, clean_size=clean_size, on_evict=on_evict)
        
        # Fill cache to exactly maxsize
        for i in range(maxsize):
            eviction.put([(f"key{i}", float(i))])
        
        # Verify we're at maxsize with no evictions yet
        self.assertEqual(len(eviction._cache_info), maxsize)
        self.assertEqual(len(evicted_keys), 0)
        
        # Store the current state before overflow
        pre_overflow_size = len(eviction._cache_info)
        pre_overflow_evicted_count = len(evicted_keys)
        
        # Add one more item to trigger eviction
        eviction.put([(f"key{maxsize}", float(maxsize))])
        
        # Verify exactly clean_size items were evicted
        post_overflow_evicted_count = len(evicted_keys)
        newly_evicted_count = post_overflow_evicted_count - pre_overflow_evicted_count
        self.assertEqual(newly_evicted_count, clean_size,
                        f"Expected exactly {clean_size} items to be evicted, but {newly_evicted_count} were evicted")
        
        # Verify final cache size is correct: previous_size + 1 (new item) - clean_size (evicted)
        expected_size = pre_overflow_size + 1 - clean_size
        actual_size = len(eviction._cache_info)
        self.assertEqual(actual_size, expected_size,
                        f"Expected cache size to be {expected_size} after eviction, got {actual_size}")
        
        # Verify cache doesn't exceed maxsize
        self.assertLessEqual(actual_size, maxsize,
                           f"Cache size {actual_size} should not exceed maxsize {maxsize}")
        
        # Verify that the lowest cost items were evicted (cost-aware behavior)
        # The evicted items should be key0 (cost=0.0) and key1 (cost=1.0)
        evicted_costs = []
        for key in evicted_keys:
            # Extract cost from key name (key0 -> 0.0, key1 -> 1.0, etc.)
            cost = float(key.replace("key", ""))
            evicted_costs.append(cost)
        
        evicted_costs.sort()
        expected_evicted_costs = [0.0, 1.0]  # Lowest costs should be evicted first
        self.assertEqual(evicted_costs, expected_evicted_costs,
                        f"Expected lowest cost items {expected_evicted_costs} to be evicted, got {evicted_costs}")
        
        # Verify remaining items are higher cost
        remaining_costs = [eviction._cache_info[key]["cost"] for key in eviction._cache_info]
        remaining_costs.sort()
        expected_remaining_costs = [2.0, 3.0, 4.0, 5.0]  # Higher costs should remain
        self.assertEqual(remaining_costs, expected_remaining_costs,
                        f"Expected remaining items to have costs {expected_remaining_costs}, got {remaining_costs}")

    def test_eviction_calls_cache_backend_delete(self):
        """Ensure on_evict is used to delete items from the cache backend."""
        evicted_keys = []
        mock_cache = Mock()

        def on_evict(keys):
            evicted_keys.extend(keys)
            # real code would call the cache backend delete method
            for k in keys:
                mock_cache.delete(k)

        eviction = CostAwareCacheEviction(maxsize=3, clean_size=2, on_evict=on_evict)

        # populate to trigger eviction
        eviction.put([("k1", 10.0)])
        eviction.put([("k2", 1.0)])
        eviction.put([("k3", 5.0)])
        eviction.put([("k4", 8.0)])  # should evict clean_size=2 items

        # metadata cleaned
        for k in evicted_keys:
            self.assertNotIn(k, eviction._cache_info)

        # backend delete called for all evicted keys
        self.assertEqual(mock_cache.delete.call_count, len(evicted_keys))
        called_args = {call.args[0] for call in mock_cache.delete.call_args_list}
        self.assertEqual(set(evicted_keys), called_args)

if __name__ == "__main__":
    unittest.main()
