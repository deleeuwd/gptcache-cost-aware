import unittest

from gptcache.manager.eviction.manager import EvictionBase


class TestCostAwareEviction(unittest.TestCase):
    def test_ca_value_is_cost(self):
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction_base = EvictionBase.get(
            name="memory", policy="ca", maxsize=3, clean_size=1, on_evict=on_evict
        )
        # Insert 3 items with different costs
        eviction_base.put([("a", 1.0), ("b", 5.5), ("c", 2.0)])

        # Retrieved value should be the stored cost
        self.assertEqual(eviction_base.get("a"), 1.0)
        self.assertEqual(eviction_base.get("b"), 5.5)
        self.assertEqual(eviction_base.get("c"), 2.0)
        self.assertEqual(evicted, [])  # no eviction yet

    def test_ca_cost_weighting_retains_expensive_item(self):
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction_base = EvictionBase.get(
            name="memory", policy="ca", maxsize=3, clean_size=1, on_evict=on_evict
        )
        # Insert items: a (cheap), b (expensive), c (cheap)
        eviction_base.put([("a", 1.0)])  # score a = 1
        eviction_base.put([("b", 10.0)])  # score b = 2
        eviction_base.put([("c", 1.0)])  # score c = 3

        # Touch expensive item once (adds +10 to its score)
        self.assertEqual(eviction_base.get("b"), 10.0)

        # Touch a cheap item several times; each touch adds only +1
        for _ in range(5):
            self.assertEqual(eviction_base.get("a"), 1.0)

        # Current logical expectations before insertion triggering eviction:
        # a score ~ 1 (base) + 5*1 = 6
        # b score ~ 2 (base) + 10 = 12
        # c score ~ 3 (base) (never touched)
        # -> c should be the lowest and evicted next.

        eviction_base.put([("d", 1.0)])  # triggers eviction of 1 key (clean_size=1)

        self.assertEqual(len(evicted), 1)
        self.assertEqual(evicted[0], "c", "Expected the lowest score key 'c' to be evicted")
        # Surviving keys should still be gettable
        self.assertIsNotNone(eviction_base.get("a"))
        self.assertIsNotNone(eviction_base.get("b"))

    def test_ca_frequency_vs_cost_tradeoff(self):
        """Cheap very hot key can eventually outrank an expensive barely-used key."""
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction_base = EvictionBase.get(
            name="memory", policy="ca", maxsize=3, clean_size=1, on_evict=on_evict
        )
        eviction_base.put([("x", 1.0)])  # score 1
        eviction_base.put([("y", 8.0)])  # score 2
        eviction_base.put([("z", 1.0)])  # score 3

        # Hammer x many times so cumulative cost overtakes y's single large cost increment.
        for _ in range(15):
            eviction_base.get("x")  # adds +1 each time
        # Do not touch y.

        # x score ~ 1 + 15 = 16, y score ~ 2, z score ~3
        # Next insertion should evict y (score 2) NOT z (score 3) because y is now lowest.
        eviction_base.put([("w", 1.0)])

        self.assertEqual(len(evicted), 1)
        self.assertEqual(evicted[0], "y", "Expected expensive but cold item 'y' to be evicted")

    def test_ca_cost_zero(self):
        """Test behavior with cost=0 (should not protect on hits)."""
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction_base = EvictionBase.get(
            name="memory", policy="ca", maxsize=3, clean_size=1, on_evict=on_evict
        )
        eviction_base.put([("a", 0.0), ("b", 1.0), ("c", 2.0)])
        # Touch a many times; since cost=0, score doesn't increase
        for _ in range(10):
            eviction_base.get("a")
        # Insert d, should evict a (score unchanged from base)
        eviction_base.put([("d", 1.0)])

        self.assertEqual(len(evicted), 1)
        self.assertEqual(evicted[0], "a")


    def test_ca_empty_cache_operations(self):
        """Test basic operations on an empty cache."""
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction_base = EvictionBase.get(
            name="memory", policy="ca", maxsize=3, clean_size=1, on_evict=on_evict
        )
        
        # Get from empty cache should return None
        self.assertIsNone(eviction_base.get("nonexistent"))
        self.assertEqual(len(evicted), 0)

    def test_ca_single_item_operations(self):
        """Test operations with a single item."""
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction_base = EvictionBase.get(
            name="memory", policy="ca", maxsize=3, clean_size=1, on_evict=on_evict
        )
        
        # Insert single item
        eviction_base.put([("single", 2.5)])
        self.assertEqual(eviction_base.get("single"), 2.5)
        
        # Touch it multiple times
        for _ in range(3):
            self.assertEqual(eviction_base.get("single"), 2.5)
        
        self.assertEqual(len(evicted), 0)

    def test_ca_maxsize_boundary(self):
        """Test behavior exactly at maxsize boundary."""
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction_base = EvictionBase.get(
            name="memory", policy="ca", maxsize=2, clean_size=1, on_evict=on_evict
        )
        
        # Fill to maxsize
        eviction_base.put([("a", 1.0), ("b", 2.0)])
        self.assertEqual(len(evicted), 0)
        
        # Adding one more should trigger eviction
        eviction_base.put([("c", 3.0)])
        self.assertEqual(len(evicted), 1)
        self.assertIn(evicted[0], ["a", "b"])  # One of them should be evicted

    def test_ca_negative_costs(self):
        """Test behavior with negative costs (edge case)."""
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction_base = EvictionBase.get(
            name="memory", policy="ca", maxsize=3, clean_size=1, on_evict=on_evict
        )
        
        eviction_base.put([("neg", -1.0), ("pos", 1.0), ("zero", 0.0)])
        
        # Negative cost should still be stored and retrievable
        self.assertEqual(eviction_base.get("neg"), -1.0)
        self.assertEqual(eviction_base.get("pos"), 1.0)
        self.assertEqual(eviction_base.get("zero"), 0.0)

    def test_ca_large_cost_values(self):
        """Test behavior with very large cost values."""
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction_base = EvictionBase.get(
            name="memory", policy="ca", maxsize=3, clean_size=1, on_evict=on_evict
        )
        
        eviction_base.put([("small", 0.001), ("large", 1000.0), ("huge", 1e6)])
        
        # All values should be stored correctly
        self.assertEqual(eviction_base.get("small"), 0.001)
        self.assertEqual(eviction_base.get("large"), 1000.0)
        self.assertEqual(eviction_base.get("huge"), 1e6)
        
        # Touch the huge cost item once - should significantly boost its score
        eviction_base.get("huge")
        
        # Add new item - small should be evicted due to lowest accumulated score
        eviction_base.put([("new", 1.0)])
        self.assertEqual(len(evicted), 1)
        self.assertEqual(evicted[0], "small")

    def test_ca_fractional_costs(self):
        """Test behavior with fractional cost values."""
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction_base = EvictionBase.get(
            name="memory", policy="ca", maxsize=3, clean_size=1, on_evict=on_evict
        )
        
        eviction_base.put([("frac1", 0.1), ("frac2", 0.25), ("frac3", 0.75)])
        
        # Touch frac2 multiple times to accumulate score
        for _ in range(10):  # 10 * 0.25 = 2.5 added to base score
            self.assertEqual(eviction_base.get("frac2"), 0.25)
        
        # Add new item - frac1 should be evicted (lowest accumulated score)
        eviction_base.put([("new", 0.5)])
        self.assertEqual(len(evicted), 1)
        self.assertEqual(evicted[0], "frac1")

    def test_ca_update_existing_key(self):
        """Test updating an existing key with new cost value."""
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction_base = EvictionBase.get(
            name="memory", policy="ca", maxsize=3, clean_size=1, on_evict=on_evict
        )
        
        # Insert initial item
        eviction_base.put([("key", 1.0)])
        self.assertEqual(eviction_base.get("key"), 1.0)
        
        # Update the same key with new cost
        eviction_base.put([("key", 5.0)])
        self.assertEqual(eviction_base.get("key"), 5.0)
        
        # No eviction should have occurred
        self.assertEqual(len(evicted), 0)

    def test_ca_mixed_access_patterns(self):
        """Test complex access patterns with different frequencies."""
        evicted = []

        def on_evict(keys):
            evicted.extend(keys)

        eviction_base = EvictionBase.get(
            name="memory", policy="ca", maxsize=4, clean_size=1, on_evict=on_evict
        )
        
        # Insert items with varying costs
        eviction_base.put([("rarely", 10.0)])     # High cost, rarely accessed
        eviction_base.put([("often", 1.0)])       # Low cost, frequently accessed
        eviction_base.put([("medium", 3.0)])      # Medium cost, medium access
        eviction_base.put([("never", 2.0)])       # Medium cost, never accessed again
        
        # Access patterns
        for _ in range(8):  # 8 * 1.0 = 8 added to base score
            eviction_base.get("often")
        
        for _ in range(2):  # 2 * 3.0 = 6 added to base score
            eviction_base.get("medium")
        
        eviction_base.get("rarely")  # 1 * 10.0 = 10 added to base score
        # "never" is never accessed again
        
        # Add new item - "never" should be evicted (lowest score)
        eviction_base.put([("new", 1.5)])
        self.assertEqual(len(evicted), 1)
        self.assertEqual(evicted[0], "never")


if __name__ == "__main__":
    unittest.main()
