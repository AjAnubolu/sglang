"""
Unit tests for the EncoderCacheManager implementation.

This module tests the scheduler-driven, ref-counted encoder output cache
introduced in issue #16957.

Test Coverage:
- Basic put/get/has operations
- Reference counting (acquire/release)
- LRU eviction of freeable entries
- Protection of active (referenced) entries from eviction
- Budget computation
- Entry lifecycle: ACTIVE -> FREEABLE -> evicted
- Size and token limit enforcement
- Multi-tier cache wrapper
- Thread safety basics
- Cache statistics

Usage:
    python test_encoder_cache_manager.py
    python -m pytest test_encoder_cache_manager.py -v
"""

from sglang.test.ci.ci_register import register_cuda_ci

# CPU-based unit test, runs quickly on any runner
register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")

import threading
import time
import unittest

import torch

from sglang.srt.mem_cache.multimodal_cache import (
    CacheEntryState,
    EncoderBudget,
    EncoderCacheEntry,
    EncoderCacheManager,
    EncoderCacheType,
    MultiTierEncoderCacheManager,
)


def _make_tensor(num_tokens: int, hidden_dim: int = 64) -> torch.Tensor:
    """Create a dummy encoder output tensor."""
    return torch.randn(num_tokens, hidden_dim)


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.element_size() * t.numel()


class TestEncoderCacheEntry(unittest.TestCase):
    """Test cases for EncoderCacheEntry dataclass."""

    def test_post_init_computes_size(self):
        t = _make_tensor(10)
        entry = EncoderCacheEntry(mm_hash=42, data=t, num_tokens=10)
        self.assertEqual(entry.data_size_bytes, _tensor_bytes(t))

    def test_default_state_is_active(self):
        entry = EncoderCacheEntry(mm_hash=1, data=_make_tensor(5), num_tokens=5)
        self.assertEqual(entry.state, CacheEntryState.ACTIVE)
        self.assertEqual(entry.ref_count, 0)


class TestEncoderCacheManager(unittest.TestCase):
    """Test cases for the core EncoderCacheManager."""

    def _make_cache(self, max_bytes=10_000_000, **kwargs):
        return EncoderCacheManager(max_size_bytes=max_bytes, **kwargs)

    # ----- Basic operations -----

    def test_put_and_get(self):
        cache = self._make_cache()
        t = _make_tensor(10)
        self.assertTrue(cache.put(1, t, num_tokens=10))
        result = cache.get(1)
        self.assertIsNotNone(result)
        self.assertTrue(torch.equal(result, t))

    def test_get_miss(self):
        cache = self._make_cache()
        self.assertIsNone(cache.get(999))

    def test_has(self):
        cache = self._make_cache()
        self.assertFalse(cache.has(1))
        cache.put(1, _make_tensor(5), num_tokens=5)
        self.assertTrue(cache.has(1))

    def test_contains_operator(self):
        cache = self._make_cache()
        cache.put(42, _make_tensor(3), num_tokens=3)
        self.assertIn(42, cache)
        self.assertNotIn(99, cache)

    def test_len(self):
        cache = self._make_cache()
        self.assertEqual(len(cache), 0)
        cache.put(1, _make_tensor(5), num_tokens=5)
        cache.put(2, _make_tensor(5), num_tokens=5)
        self.assertEqual(len(cache), 2)

    def test_clear(self):
        cache = self._make_cache()
        cache.put(1, _make_tensor(5), num_tokens=5)
        cache.put(2, _make_tensor(5), num_tokens=5)
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertFalse(cache.has(1))
        self.assertEqual(cache.current_size_bytes, 0)

    # ----- Reference counting -----

    def test_acquire_increments_refcount(self):
        cache = self._make_cache()
        cache.put(1, _make_tensor(5), num_tokens=5)
        result = cache.acquire(1)
        self.assertIsNotNone(result)
        self.assertEqual(cache.num_active, 1)
        self.assertEqual(cache.num_freeable, 0)

    def test_acquire_miss_returns_none(self):
        cache = self._make_cache()
        self.assertIsNone(cache.acquire(999))

    def test_release_decrements_refcount(self):
        cache = self._make_cache()
        cache.put(1, _make_tensor(5), num_tokens=5)
        cache.acquire(1)
        self.assertEqual(cache.num_active, 1)
        cache.release(1)
        self.assertEqual(cache.num_active, 0)
        self.assertEqual(cache.num_freeable, 1)

    def test_multiple_acquires_and_releases(self):
        cache = self._make_cache()
        cache.put(1, _make_tensor(5), num_tokens=5)
        cache.acquire(1)
        cache.acquire(1)
        self.assertEqual(cache.num_active, 1)
        # Release once - still active (ref_count=1)
        cache.release(1)
        self.assertEqual(cache.num_active, 1)
        # Release again - now freeable (ref_count=0)
        cache.release(1)
        self.assertEqual(cache.num_active, 0)
        self.assertEqual(cache.num_freeable, 1)

    def test_acquire_promotes_from_freeable(self):
        cache = self._make_cache()
        cache.put(1, _make_tensor(5), num_tokens=5)
        # Entry starts as FREEABLE (ref_count=0 after put)
        self.assertEqual(cache.num_freeable, 1)
        # Acquire promotes to ACTIVE
        cache.acquire(1)
        self.assertEqual(cache.num_active, 1)
        self.assertEqual(cache.num_freeable, 0)

    # ----- Eviction -----

    def test_evict_freeable_entry(self):
        cache = self._make_cache()
        cache.put(1, _make_tensor(5), num_tokens=5)
        # Entry is freeable (no references)
        self.assertTrue(cache.evict(1))
        self.assertFalse(cache.has(1))

    def test_cannot_evict_active_entry(self):
        cache = self._make_cache()
        cache.put(1, _make_tensor(5), num_tokens=5)
        cache.acquire(1)
        # Cannot evict while referenced
        self.assertFalse(cache.evict(1))
        self.assertTrue(cache.has(1))

    def test_lru_eviction_on_put(self):
        """When cache is full, freeable entries are evicted in LRU order."""
        t1 = _make_tensor(10)
        t2 = _make_tensor(10)
        size = _tensor_bytes(t1)
        # Cache can hold exactly one entry
        cache = self._make_cache(max_bytes=size)
        cache.put(1, t1, num_tokens=10)
        self.assertTrue(cache.has(1))
        # Inserting second entry should evict first
        cache.put(2, t2, num_tokens=10)
        self.assertFalse(cache.has(1))
        self.assertTrue(cache.has(2))

    def test_active_entries_not_evicted_on_put(self):
        """Active (referenced) entries survive eviction pressure."""
        t1 = _make_tensor(10)
        t2 = _make_tensor(10)
        size = _tensor_bytes(t1)
        cache = self._make_cache(max_bytes=size)
        cache.put(1, t1, num_tokens=10)
        cache.acquire(1)  # Protect entry 1
        # Cannot insert t2 because entry 1 is active and can't be evicted
        self.assertFalse(cache.put(2, t2, num_tokens=10))
        self.assertTrue(cache.has(1))
        self.assertFalse(cache.has(2))

    def test_evict_freeable_reclaims_space(self):
        cache = self._make_cache()
        t1 = _make_tensor(100)
        cache.put(1, t1, num_tokens=100)
        size_before = cache.current_size_bytes
        freed = cache.evict_freeable(target_bytes=0)
        self.assertEqual(freed, size_before)
        self.assertEqual(cache.current_size_bytes, 0)

    def test_evict_freeable_with_target(self):
        cache = self._make_cache()
        t1 = _make_tensor(10)
        t2 = _make_tensor(10)
        cache.put(1, t1, num_tokens=10)
        cache.put(2, t2, num_tokens=10)
        # Evict just enough to free one entry
        freed = cache.evict_freeable(target_bytes=_tensor_bytes(t1))
        self.assertGreaterEqual(freed, _tensor_bytes(t1))
        # At least one entry should remain
        self.assertLessEqual(len(cache), 2)

    # ----- Size limits -----

    def test_max_entry_count(self):
        cache = self._make_cache(max_entry_count=2)
        cache.put(1, _make_tensor(5), num_tokens=5)
        cache.put(2, _make_tensor(5), num_tokens=5)
        # Third entry evicts the oldest freeable
        self.assertTrue(cache.put(3, _make_tensor(5), num_tokens=5))
        self.assertEqual(len(cache), 2)
        # Entry 1 (oldest) should have been evicted
        self.assertFalse(cache.has(1))

    def test_max_encoder_tokens(self):
        cache = self._make_cache(max_encoder_tokens=20)
        cache.put(1, _make_tensor(10), num_tokens=10)
        cache.put(2, _make_tensor(10), num_tokens=10)
        # Cache is at 20 tokens, adding more should evict
        self.assertTrue(cache.put(3, _make_tensor(5), num_tokens=5))
        # Total should not exceed 20 after eviction
        budget = cache.compute_budget()
        self.assertLessEqual(budget.used_tokens, 20)

    def test_put_fails_when_single_entry_exceeds_max(self):
        t = _make_tensor(100)
        size = _tensor_bytes(t)
        # Max size is smaller than a single entry
        cache = self._make_cache(max_bytes=size - 1)
        self.assertFalse(cache.put(1, t, num_tokens=100))

    # ----- Update existing entry -----

    def test_put_updates_existing(self):
        cache = self._make_cache()
        t1 = _make_tensor(10, hidden_dim=32)
        t2 = _make_tensor(10, hidden_dim=64)
        cache.put(1, t1, num_tokens=10)
        size1 = cache.current_size_bytes
        cache.put(1, t2, num_tokens=10)
        self.assertEqual(len(cache), 1)
        result = cache.get(1)
        self.assertTrue(torch.equal(result, t2))
        # Size should reflect new tensor
        self.assertEqual(cache.current_size_bytes, _tensor_bytes(t2))

    # ----- Budget computation -----

    def test_compute_budget_empty(self):
        cache = self._make_cache(max_bytes=1000, max_encoder_tokens=50)
        budget = cache.compute_budget()
        self.assertEqual(budget.remaining_bytes, 1000)
        self.assertEqual(budget.remaining_tokens, 50)
        self.assertEqual(budget.freeable_bytes, 0)
        self.assertEqual(budget.freeable_tokens, 0)
        self.assertTrue(budget.can_admit(500, 25))

    def test_compute_budget_after_inserts(self):
        cache = self._make_cache(max_bytes=100_000, max_encoder_tokens=100)
        t = _make_tensor(30)
        cache.put(1, t, num_tokens=30)
        budget = cache.compute_budget()
        self.assertEqual(budget.used_tokens, 30)
        self.assertEqual(budget.used_bytes, _tensor_bytes(t))
        self.assertEqual(budget.remaining_tokens, 70)
        self.assertEqual(budget.freeable_tokens, 30)  # entry is freeable
        self.assertTrue(budget.can_admit(_tensor_bytes(t), 30))

    def test_budget_can_admit_with_freeable(self):
        t = _make_tensor(10)
        size = _tensor_bytes(t)
        cache = self._make_cache(max_bytes=size)
        cache.put(1, t, num_tokens=10)
        budget = cache.compute_budget()
        # Remaining is 0, but freeable covers it
        self.assertEqual(budget.remaining_bytes, 0)
        self.assertEqual(budget.available_bytes, size)
        self.assertTrue(budget.can_admit(size, 10))

    def test_budget_cannot_admit_when_active_fills_cache(self):
        t = _make_tensor(10)
        size = _tensor_bytes(t)
        cache = self._make_cache(max_bytes=size)
        cache.put(1, t, num_tokens=10)
        cache.acquire(1)  # Make it active
        budget = cache.compute_budget()
        # Nothing freeable
        self.assertEqual(budget.freeable_bytes, 0)
        self.assertFalse(budget.can_admit(size, 10))

    # ----- Statistics -----

    def test_hit_rate(self):
        cache = self._make_cache()
        cache.put(1, _make_tensor(5), num_tokens=5)
        cache.get(1)  # hit
        cache.get(2)  # miss
        self.assertAlmostEqual(cache.hit_rate, 0.5)

    def test_get_stats(self):
        cache = self._make_cache()
        cache.put(1, _make_tensor(5), num_tokens=5)
        cache.acquire(1)
        stats = cache.get_stats()
        self.assertEqual(stats["num_entries"], 1)
        self.assertEqual(stats["num_active"], 1)
        self.assertEqual(stats["num_freeable"], 0)

    # ----- Combined hash lookup -----

    def test_get_by_combined_hash(self):
        cache = self._make_cache()
        from sglang.srt.mem_cache.multimodal_cache import MultimodalCache

        combined = MultimodalCache.combine_hashes([10, 20, 30])
        t = _make_tensor(15)
        cache.put(combined, t, num_tokens=15)
        result = cache.get_by_combined_hash([10, 20, 30])
        self.assertIsNotNone(result)
        self.assertTrue(torch.equal(result, t))

    def test_get_by_combined_hash_empty_list(self):
        cache = self._make_cache()
        self.assertIsNone(cache.get_by_combined_hash([]))

    # ----- Thread safety -----

    def test_concurrent_access(self):
        """Basic thread safety test: multiple threads doing put/get."""
        cache = self._make_cache()
        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    key = thread_id * 1000 + i
                    t = _make_tensor(5)
                    cache.put(key, t, num_tokens=5)
                    cache.acquire(key)
                    result = cache.get(key)
                    if result is None:
                        errors.append(f"Thread {thread_id}: get({key}) returned None")
                    cache.release(key)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], f"Errors: {errors}")


class TestEncoderBudget(unittest.TestCase):
    """Test cases for EncoderBudget."""

    def test_available_bytes(self):
        budget = EncoderBudget(
            remaining_bytes=100,
            remaining_tokens=50,
            remaining_entries=None,
            freeable_bytes=200,
            freeable_tokens=30,
            total_bytes=1000,
            used_bytes=700,
            total_tokens=100,
            used_tokens=20,
        )
        self.assertEqual(budget.available_bytes, 300)
        self.assertEqual(budget.available_tokens, 80)

    def test_available_tokens_none_when_unlimited(self):
        budget = EncoderBudget(
            remaining_bytes=100,
            remaining_tokens=None,
            remaining_entries=None,
            freeable_bytes=200,
            freeable_tokens=30,
            total_bytes=1000,
            used_bytes=700,
            total_tokens=None,
            used_tokens=0,
        )
        self.assertIsNone(budget.available_tokens)

    def test_can_admit(self):
        budget = EncoderBudget(
            remaining_bytes=0,
            remaining_tokens=0,
            remaining_entries=None,
            freeable_bytes=500,
            freeable_tokens=50,
            total_bytes=1000,
            used_bytes=1000,
            total_tokens=100,
            used_tokens=100,
        )
        self.assertTrue(budget.can_admit(500, 50))
        self.assertFalse(budget.can_admit(501, 50))
        self.assertFalse(budget.can_admit(500, 51))


class TestMultiTierEncoderCacheManager(unittest.TestCase):
    """Test cases for the two-tier cache wrapper."""

    def test_separate_tiers(self):
        mtier = MultiTierEncoderCacheManager(
            feature_cache_bytes=10_000_000,
            embedding_cache_bytes=10_000_000,
        )
        t_feat = _make_tensor(10)
        t_emb = _make_tensor(10)

        mtier.feature_cache.put(1, t_feat, num_tokens=10)
        mtier.embedding_cache.put(1, t_emb, num_tokens=10)

        # Same key in different tiers should give different tensors
        feat = mtier.feature_cache.get(1)
        emb = mtier.embedding_cache.get(1)
        self.assertTrue(torch.equal(feat, t_feat))
        self.assertTrue(torch.equal(emb, t_emb))

    def test_get_cache_by_type(self):
        mtier = MultiTierEncoderCacheManager(
            feature_cache_bytes=1000,
            embedding_cache_bytes=2000,
        )
        fc = mtier.get_cache(EncoderCacheType.FEATURE)
        ec = mtier.get_cache(EncoderCacheType.EMBEDDING)
        self.assertIsInstance(fc, EncoderCacheManager)
        self.assertIsInstance(ec, EncoderCacheManager)
        self.assertIsNot(fc, ec)

    def test_clear_both_tiers(self):
        mtier = MultiTierEncoderCacheManager(
            feature_cache_bytes=10_000_000,
            embedding_cache_bytes=10_000_000,
        )
        mtier.feature_cache.put(1, _make_tensor(5), num_tokens=5)
        mtier.embedding_cache.put(1, _make_tensor(5), num_tokens=5)
        mtier.clear()
        self.assertEqual(len(mtier.feature_cache), 0)
        self.assertEqual(len(mtier.embedding_cache), 0)

    def test_get_stats(self):
        mtier = MultiTierEncoderCacheManager(
            feature_cache_bytes=10_000_000,
            embedding_cache_bytes=10_000_000,
        )
        mtier.feature_cache.put(1, _make_tensor(5), num_tokens=5)
        stats = mtier.get_stats()
        self.assertIn("mm_feature", stats)
        self.assertIn("mm_embedding", stats)
        self.assertEqual(stats["mm_feature"]["num_entries"], 1)
        self.assertEqual(stats["mm_embedding"]["num_entries"], 0)


class TestEncoderCacheManagerLRUOrder(unittest.TestCase):
    """Test that LRU ordering works correctly for eviction."""

    def test_lru_eviction_order(self):
        """Oldest freeable entry is evicted first."""
        t = _make_tensor(10)
        size = _tensor_bytes(t)
        # Can hold exactly 2 entries
        cache = EncoderCacheManager(max_size_bytes=size * 2)
        cache.put(1, _make_tensor(10), num_tokens=10)
        cache.put(2, _make_tensor(10), num_tokens=10)
        # Access entry 1 to make it more recent
        cache.get(1)
        # Insert third entry, should evict entry 2 (least recently used)
        cache.put(3, _make_tensor(10), num_tokens=10)
        self.assertFalse(cache.has(2))  # evicted
        self.assertTrue(cache.has(1))   # kept (more recent)
        self.assertTrue(cache.has(3))   # just inserted

    def test_acquire_updates_lru(self):
        """Acquiring an entry should update its LRU position."""
        t = _make_tensor(10)
        size = _tensor_bytes(t)
        cache = EncoderCacheManager(max_size_bytes=size * 2)
        cache.put(1, _make_tensor(10), num_tokens=10)
        cache.put(2, _make_tensor(10), num_tokens=10)
        # Acquire entry 1 (makes it most recent + active)
        cache.acquire(1)
        cache.release(1)
        # Insert third entry, should evict entry 2 (LRU freeable)
        cache.put(3, _make_tensor(10), num_tokens=10)
        self.assertTrue(cache.has(1))
        self.assertFalse(cache.has(2))


class TestEncoderCacheManagerEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_release_nonexistent(self):
        cache = EncoderCacheManager(max_size_bytes=10000)
        self.assertFalse(cache.release(999))

    def test_evict_nonexistent(self):
        cache = EncoderCacheManager(max_size_bytes=10000)
        self.assertFalse(cache.evict(999))

    def test_double_release(self):
        """Releasing more times than acquired should not go negative."""
        cache = EncoderCacheManager(max_size_bytes=10_000_000)
        cache.put(1, _make_tensor(5), num_tokens=5)
        cache.acquire(1)
        cache.release(1)
        cache.release(1)  # Extra release
        self.assertEqual(cache.num_freeable, 1)
        self.assertEqual(cache.num_active, 0)

    def test_empty_cache_stats(self):
        cache = EncoderCacheManager(max_size_bytes=1000)
        stats = cache.get_stats()
        self.assertEqual(stats["num_entries"], 0)
        self.assertEqual(stats["current_size_bytes"], 0)
        self.assertAlmostEqual(cache.hit_rate, 0.0)

    def test_put_zero_token_entry(self):
        cache = EncoderCacheManager(max_size_bytes=10_000_000)
        # A 1D tensor with 0 tokens
        t = torch.empty(0, 64)
        self.assertTrue(cache.put(1, t, num_tokens=0))
        self.assertTrue(cache.has(1))

    def test_evict_freeable_on_empty_cache(self):
        cache = EncoderCacheManager(max_size_bytes=1000)
        freed = cache.evict_freeable(target_bytes=100)
        self.assertEqual(freed, 0)


if __name__ == "__main__":
    unittest.main()
