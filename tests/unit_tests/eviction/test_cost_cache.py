import unittest
import math
from unittest.mock import patch

from gptcache.manager.eviction.cost_cache import CACache


class TestCACache(unittest.TestCase):
    """Unit tests for Cost-Aware Cache (CACache) implementation."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.cache = CACache(maxsize=3, tau=100)

    def test_init(self):
        """Test cache initialization with default and custom parameters."""
        # Test default initialization
        cache = CACache(maxsize=5)
        self.assertEqual(cache.maxsize, 5)
        self.assertIsInstance(cache._tau, (int, float))  # tau should be a number
        self.assertGreater(cache._tau, 0)  # tau should be positive
        self.assertEqual(len(cache._scores), 0)
        self.assertEqual(len(cache._costs), 0)
        self.assertEqual(len(cache._freqs), 0)
        self.assertEqual(len(cache._gen), 0)
        self.assertEqual(len(cache._heap), 0)

        # Test initialization with custom tau
        cache_custom = CACache(maxsize=10, tau=50)
        self.assertEqual(cache_custom.maxsize, 10)
        self.assertEqual(cache_custom._tau, 50)

    def test_setitem_and_getitem(self):
        """Test setting and getting items with costs."""
        # Test inserting new item
        self.cache.__setitem__("key1", 1.0)
        self.assertIn("key1", self.cache)
        self.assertEqual(self.cache._costs["key1"], 1.0)
        self.assertEqual(self.cache._freqs["key1"], 1)
        self.assertEqual(len(self.cache._heap), 1)

        # Test getting item (should increase frequency)
        value = self.cache.__getitem__("key1")
        self.assertEqual(value, 1.0)
        self.assertEqual(self.cache._freqs["key1"], 2)
        
        # Test updating existing item with new cost
        self.cache.__setitem__("key1", 2.0)
        self.assertEqual(self.cache._costs["key1"], 2.0)
        self.assertEqual(self.cache._freqs["key1"], 3)  # frequency should increase

    def test_delitem(self):
        """Test deleting items from cache."""
        # Add item
        self.cache.__setitem__("key1", 1.0)
        self.assertIn("key1", self.cache)
        
        # Delete item
        self.cache.__delitem__("key1")
        self.assertNotIn("key1", self.cache)
        self.assertNotIn("key1", self.cache._scores)
        self.assertNotIn("key1", self.cache._costs)
        self.assertNotIn("key1", self.cache._freqs)
        self.assertNotIn("key1", self.cache._gen)

    def test_score_calculation(self):
        """Test that scores are calculated correctly using the formula: tau * cost * log(1 + freq)."""
        # Insert item with cost 2.0
        self.cache.__setitem__("key1", 2.0)
        
        # Initial score should be tau * cost * log(1 + 1) = tau * 2.0 * log(2)
        expected_score = self.cache._tau * 2.0 * math.log1p(1)
        self.assertAlmostEqual(self.cache._scores["key1"], expected_score, places=10)
        
        # Access the item to increase frequency
        self.cache.__getitem__("key1")
        
        # New score should be tau * cost * log(1 + 2) = tau * 2.0 * log(3)
        expected_score = self.cache._tau * 2.0 * math.log1p(2)
        self.assertAlmostEqual(self.cache._scores["key1"], expected_score, places=10)

    def test_eviction_order(self):
        """Test that items with lowest scores are evicted first."""
        # Add items with different costs
        self.cache.__setitem__("low_cost", 1.0)    # Lowest cost = lowest initial score
        self.cache.__setitem__("high_cost", 5.0)   # Highest cost = highest initial score
        self.cache.__setitem__("med_cost", 3.0)    # Medium cost = medium initial score
        
        # Cache is now full (maxsize=3)
        self.assertEqual(len(self.cache), 3)
        
        # Add another item, should evict the lowest score (low_cost)
        self.cache.__setitem__("new_item", 2.0)
        
        # Verify eviction
        self.assertEqual(len(self.cache), 3)
        self.assertNotIn("low_cost", self.cache)
        self.assertIn("high_cost", self.cache)
        self.assertIn("med_cost", self.cache)
        self.assertIn("new_item", self.cache)

    def test_frequency_affects_eviction(self):
        """Test that frequently accessed items are less likely to be evicted."""
        # Add items with same cost
        self.cache.__setitem__("item1", 1.0)
        self.cache.__setitem__("item2", 1.0)
        self.cache.__setitem__("item3", 1.0)
        
        # Access item1 multiple times to increase its frequency
        for _ in range(5):
            self.cache.__getitem__("item1")
        
        # Add new item, should evict item with lowest score (item2 or item3)
        self.cache.__setitem__("new_item", 1.0)
        
        # item1 should still be in cache due to higher frequency
        self.assertIn("item1", self.cache)
        self.assertIn("new_item", self.cache)
        self.assertEqual(len(self.cache), 3)

    def test_popitem_empty_cache(self):
        """Test that popitem raises KeyError on empty cache."""
        with self.assertRaises(KeyError) as cm:
            self.cache.popitem()
        
        self.assertIn("CACache is empty", str(cm.exception))

    def test_popitem_returns_lowest_score_item(self):
        """Test that popitem returns the item with the lowest score."""
        # Add items with different costs
        self.cache.__setitem__("high_cost", 5.0)
        self.cache.__setitem__("low_cost", 1.0)
        self.cache.__setitem__("med_cost", 3.0)
        
        # Pop item should return the one with lowest score
        key, value = self.cache.popitem()
        self.assertEqual(key, "low_cost")
        self.assertEqual(value, 1.0)
        self.assertNotIn("low_cost", self.cache)
        self.assertEqual(len(self.cache), 2)

    def test_heap_consistency(self):
        """Test that heap maintains consistency after multiple operations."""
        # Add items within cache capacity to avoid eviction complications
        cache = CACache(maxsize=10, tau=100)  # Use larger cache for this test
        
        # Add items
        for i in range(5):
            cache.__setitem__(f"key{i}", float(i + 1))
            
        # Access some items multiple times
        for i in [1, 2, 3]:
            for _ in range(3):
                cache.__getitem__(f"key{i}")
        
        # Verify heap has valid entries
        valid_heap_entries = 0
        for score, gen, key in cache._heap:
            if key in cache._scores and cache._gen.get(key) == gen:
                valid_heap_entries += 1
        
        self.assertGreaterEqual(valid_heap_entries, len(cache))

    def test_cost_type_conversion(self):
        """Test that costs are properly converted to floats."""
        # Test with integer cost
        self.cache.__setitem__("int_cost", 5)
        self.assertEqual(self.cache._costs["int_cost"], 5.0)
        self.assertIsInstance(self.cache._costs["int_cost"], float)
        
        # Test with string cost (should convert)
        self.cache.__setitem__("str_cost", "3.5")
        self.assertEqual(self.cache._costs["str_cost"], 3.5)
        self.assertIsInstance(self.cache._costs["str_cost"], float)

    def test_tau_parameter_effect(self):
        """Test that tau parameter affects score calculation."""
        cache1 = CACache(maxsize=2, tau=100)
        cache2 = CACache(maxsize=2, tau=200)
        
        cache1.__setitem__("key", 2.0)
        cache2.__setitem__("key", 2.0)
        
        # Scores should be different due to different tau values
        score1 = cache1._scores["key"]
        score2 = cache2._scores["key"]
        
        self.assertAlmostEqual(score2, 2 * score1, places=10)

    def test_generation_counter(self):
        """Test that generation counter is properly maintained."""
        self.cache.__setitem__("key1", 1.0)
        initial_gen = self.cache._gen["key1"]
        
        # Access item multiple times
        for i in range(3):
            self.cache.__getitem__("key1")
            self.assertEqual(self.cache._gen["key1"], initial_gen + i + 1)

    def test_multiple_items_same_score(self):
        """Test behavior when multiple items have the same score."""
        # Add items with same cost (will have same initial score)
        self.cache.__setitem__("key1", 2.0)
        self.cache.__setitem__("key2", 2.0)
        self.cache.__setitem__("key3", 2.0)
        
        # Fill cache and add one more
        self.cache.__setitem__("key4", 2.0)
        
        # Should evict one of the items (likely the first one due to generation counter)
        self.assertEqual(len(self.cache), 3)
        self.assertNotIn("key1", self.cache)

    def test_cache_integration_with_cachetools(self):
        """Test that CACache properly inherits from cachetools.Cache."""
        # Test basic cache operations work
        self.cache["test_key"] = 1.5
        self.assertEqual(self.cache["test_key"], 1.5)
        self.assertTrue("test_key" in self.cache)
        
        # Test cache size limits
        for i in range(5):  # maxsize is 3
            self.cache[f"key_{i}"] = float(i)
        
        self.assertEqual(len(self.cache), 3)

    def test_edge_case_zero_cost(self):
        """Test handling of zero cost items."""
        self.cache.__setitem__("zero_cost", 0.0)
        self.assertEqual(self.cache._costs["zero_cost"], 0.0)
        self.assertEqual(self.cache._scores["zero_cost"], 0.0)  # tau * 0 * log(2) = 0

    def test_edge_case_negative_cost(self):
        """Test handling of negative cost items."""
        self.cache.__setitem__("neg_cost", -1.0)
        self.assertEqual(self.cache._costs["neg_cost"], -1.0)
        # Score should be negative: tau * (-1.0) * log(2)
        expected_score = self.cache._tau * (-1.0) * math.log1p(1)
        self.assertAlmostEqual(self.cache._scores["neg_cost"], expected_score, places=10)


if __name__ == "__main__":
    unittest.main()