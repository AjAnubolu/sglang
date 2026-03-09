import abc
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)


class MultimodalCache(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
    ): ...

    @staticmethod
    def combine_hashes(mm_hashes: List[int]) -> Optional[int]:
        """
        Get a combined hash from individual mm item hashes
        """
        if not mm_hashes:
            return None
        return hash(tuple(mm_hashes))

    @abc.abstractmethod
    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        Extract the embedding with the hash-ids of the queried items. Try combined hash first, if missed, fallback to individual hashes
        The returned tensor may not be contiguous
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set(
        self,
        mm_hash: int,
        embedding: torch.Tensor,
        mm_embedding_allocator: BaseTokenToKVPoolAllocator,
    ) -> bool:
        """
        Set the embedding to the pre-allocated locations with a hash id
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def has(self, mm_hash: int) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def available_size(self):
        raise NotImplementedError()


def _get_tensor_size(embedding: torch.Tensor):
    return embedding.element_size() * embedding.numel()


@dataclass(kw_only=True)
class EmbeddingResult:
    embedding: torch.Tensor


class MultiModalStaticCache(MultimodalCache):
    """
    A server-level cache for multimodal embedding.
    Embeddings are computed prior, and this cache does not really pre-alloc
    """

    def __init__(
        self,
        max_size: int,
    ):
        super().__init__()
        self.max_size = max_size
        self.mm_cache: OrderedDict[int, EmbeddingResult] = OrderedDict()
        self.current_size = 0

    def get(
        self, mm_hashes: List[int], combined_hash: Optional[int] = None
    ) -> Optional[EmbeddingResult]:
        combined_hash = self.combine_hashes(mm_hashes)
        # MultiModalStaticCache does not fallback to individual item lookup

        embedding = self.mm_cache.get(combined_hash)
        if embedding is not None:
            self.mm_cache.move_to_end(combined_hash)
        return embedding

    def set(
        self,
        mm_hash: int,
        embedding: EmbeddingResult,
        loc: Optional[torch.Tensor] = None,
    ) -> bool:
        assert isinstance(embedding, EmbeddingResult), embedding
        if mm_hash in self.mm_cache:
            self.mm_cache.move_to_end(mm_hash)
            return True
        data_size = _get_tensor_size(embedding.embedding)
        while self.current_size + data_size > self.max_size:
            if not self.mm_cache:
                return False
            lru_hash, lru_embedding = self.mm_cache.popitem(last=False)
            self.current_size -= _get_tensor_size(lru_embedding.embedding)

        self.mm_cache[mm_hash] = embedding
        self.current_size += data_size
        return True

    def has(self, mm_hash: int) -> bool:
        return mm_hash in self.mm_cache

    def free(
        self, mm_hash: int, mm_embedding_allocator: BaseTokenToKVPoolAllocator
    ) -> bool:
        if mm_hash not in self.mm_cache:
            return False
        old_embedding = self.mm_cache.pop(mm_hash)
        self.current_size -= _get_tensor_size(old_embedding.embedding)
        return True

    def clear(self):
        self.mm_cache.clear()
        self.current_size = 0

    def __len__(self):
        return len(self.mm_cache)

    def available_size(self):
        return self.__len__()


# ---------------------------------------------------------------------------
# EncoderCacheManager: scheduler-driven, ref-counted encoder output cache
# ---------------------------------------------------------------------------


class CacheEntryState(Enum):
    """Lifecycle states for a cache entry."""

    ACTIVE = auto()  # In use by at least one request
    FREEABLE = auto()  # No active references, eligible for eviction
    FREED = auto()  # Evicted, memory released


@dataclass
class EncoderCacheEntry:
    """A single cached encoder output with reference tracking.

    Attributes:
        mm_hash: The hash key identifying this multimodal input.
        data: The cached tensor (encoder feature or embedding).
        num_tokens: Number of encoder output tokens this entry represents.
        ref_count: Number of active requests using this entry.
        state: Current lifecycle state.
        last_access_time: Monotonic timestamp of last access (for LRU).
        creation_time: Monotonic timestamp of creation.
        data_size_bytes: Size of the cached tensor in bytes.
    """

    mm_hash: int
    data: torch.Tensor
    num_tokens: int
    ref_count: int = 0
    state: CacheEntryState = CacheEntryState.ACTIVE
    last_access_time: float = field(default_factory=time.monotonic)
    creation_time: float = field(default_factory=time.monotonic)
    data_size_bytes: int = 0

    def __post_init__(self):
        if self.data is not None:
            self.data_size_bytes = _get_tensor_size(self.data)


class EncoderCacheManager:
    """Scheduler-driven cache manager for encoder outputs (vision features and
    multimodal embeddings).

    Unlike the passive LRU ``MultiModalStaticCache``, this manager is
    **actively driven by the scheduler** with reference counting and a
    three-state lifecycle (ACTIVE -> FREEABLE -> FREED).

    Design contract (from issue #16957):
      Decisions on allocation and eviction are made during the scheduling
      phase.  The actual physical release is completed by the worker at an
      appropriate time.

    Key capabilities:
      - **Reference counting**: entries are not evicted while any request
        holds a reference.
      - **Recyclable queue**: entries move ACTIVE -> FREEABLE when ref_count
        drops to zero, and can be promoted back to ACTIVE on a cache hit.
      - **Budget computation**: ``compute_budget`` reports how many encoder
        tokens can still be admitted given the current memory usage.
      - **Two-tier caching**: supports separate namespaces for raw encoder
        features (``mm_feature``) and projected embeddings (``mm_embedding``).
      - **Thread safety**: all public methods are protected by a lock so
        the manager can be shared between the scheduler and workers.

    Parameters:
        max_size_bytes: Maximum total size in bytes for cached tensors.
        max_entry_count: Optional cap on the number of entries.
        max_encoder_tokens: Optional cap on total encoder tokens across all
            cached entries.
    """

    def __init__(
        self,
        max_size_bytes: int,
        max_entry_count: Optional[int] = None,
        max_encoder_tokens: Optional[int] = None,
    ):
        self._max_size_bytes = max_size_bytes
        self._max_entry_count = max_entry_count
        self._max_encoder_tokens = max_encoder_tokens
        self._lock = threading.Lock()

        # Primary cache: mm_hash -> EncoderCacheEntry
        self._cache: OrderedDict[int, EncoderCacheEntry] = OrderedDict()

        # Tracking sets for lifecycle management
        self._active_hashes: Set[int] = set()
        self._freeable_hashes: Set[int] = set()

        # Accounting
        self._current_size_bytes: int = 0
        self._current_encoder_tokens: int = 0
        self._hit_count: int = 0
        self._miss_count: int = 0

    # ------------------------------------------------------------------
    # Public API – cache operations
    # ------------------------------------------------------------------

    def get(self, mm_hash: int) -> Optional[torch.Tensor]:
        """Look up a cached encoder output by hash.

        If found, the entry is touched (moved to end of LRU order) and
        returned.  This does **not** increment the reference count -- use
        ``acquire`` for that.

        Returns:
            The cached tensor, or ``None`` on a miss.
        """
        with self._lock:
            entry = self._cache.get(mm_hash)
            if entry is None or entry.state == CacheEntryState.FREED:
                self._miss_count += 1
                return None
            self._hit_count += 1
            entry.last_access_time = time.monotonic()
            self._cache.move_to_end(mm_hash)
            return entry.data

    def get_by_combined_hash(
        self, mm_hashes: List[int]
    ) -> Optional[torch.Tensor]:
        """Look up using a combined hash derived from a list of item hashes.

        Mirrors the ``MultiModalStaticCache.combine_hashes`` convention so
        callers can use either individual or combined keys.
        """
        combined = MultimodalCache.combine_hashes(mm_hashes)
        if combined is None:
            return None
        return self.get(combined)

    def put(
        self,
        mm_hash: int,
        data: torch.Tensor,
        num_tokens: int = 0,
    ) -> bool:
        """Insert or update an encoder output in the cache.

        If the hash already exists, the entry is refreshed (data replaced,
        access time updated) without changing the reference count.

        If insertion requires eviction and there is not enough freeable
        space, the call returns ``False``.

        Args:
            mm_hash: Hash key for the multimodal input.
            data: Encoder output tensor to cache.
            num_tokens: Number of encoder output tokens this entry covers.

        Returns:
            ``True`` if the entry was successfully cached.
        """
        with self._lock:
            return self._put_locked(mm_hash, data, num_tokens)

    def has(self, mm_hash: int) -> bool:
        """Check whether a hash is present and not freed."""
        with self._lock:
            entry = self._cache.get(mm_hash)
            return entry is not None and entry.state != CacheEntryState.FREED

    def acquire(self, mm_hash: int) -> Optional[torch.Tensor]:
        """Acquire a reference to a cached entry.

        Increments the reference count and ensures the entry is in ACTIVE
        state (promoting from FREEABLE if necessary).  This should be called
        by the **scheduler** when a request begins using a cached encoder
        output.

        Returns:
            The cached tensor, or ``None`` if the hash is not in the cache.
        """
        with self._lock:
            entry = self._cache.get(mm_hash)
            if entry is None or entry.state == CacheEntryState.FREED:
                return None
            entry.ref_count += 1
            entry.last_access_time = time.monotonic()
            if entry.state == CacheEntryState.FREEABLE:
                self._freeable_hashes.discard(mm_hash)
                entry.state = CacheEntryState.ACTIVE
                self._active_hashes.add(mm_hash)
            self._cache.move_to_end(mm_hash)
            return entry.data

    def release(self, mm_hash: int) -> bool:
        """Release a reference to a cached entry.

        Decrements the reference count.  When the count reaches zero the
        entry transitions to FREEABLE and becomes eligible for eviction.

        This should be called by the **scheduler** when a request that was
        using this encoder output finishes or is preempted.

        Returns:
            ``True`` if the release was successful.
        """
        with self._lock:
            entry = self._cache.get(mm_hash)
            if entry is None or entry.state == CacheEntryState.FREED:
                return False
            entry.ref_count = max(0, entry.ref_count - 1)
            if entry.ref_count == 0 and entry.state == CacheEntryState.ACTIVE:
                entry.state = CacheEntryState.FREEABLE
                self._active_hashes.discard(mm_hash)
                self._freeable_hashes.add(mm_hash)
            return True

    def evict(self, mm_hash: int) -> bool:
        """Explicitly evict a single entry.

        Only FREEABLE entries (ref_count == 0) can be evicted.

        Returns:
            ``True`` if the entry was evicted.
        """
        with self._lock:
            return self._evict_entry_locked(mm_hash)

    def evict_freeable(self, target_bytes: int = 0) -> int:
        """Evict FREEABLE entries in LRU order until at least
        ``target_bytes`` of memory has been reclaimed (or all freeable
        entries have been evicted).

        Args:
            target_bytes: Minimum number of bytes to free.  Pass 0 to evict
                all freeable entries.

        Returns:
            Total bytes actually freed.
        """
        with self._lock:
            return self._evict_freeable_locked(target_bytes)

    def clear(self) -> None:
        """Remove all entries regardless of state."""
        with self._lock:
            self._cache.clear()
            self._active_hashes.clear()
            self._freeable_hashes.clear()
            self._current_size_bytes = 0
            self._current_encoder_tokens = 0

    # ------------------------------------------------------------------
    # Public API – budget and introspection
    # ------------------------------------------------------------------

    def compute_budget(self) -> "EncoderBudget":
        """Compute how many more encoder tokens and bytes the cache can
        accept before eviction is required.

        Returns an ``EncoderBudget`` with the remaining capacity.
        """
        with self._lock:
            remaining_bytes = max(0, self._max_size_bytes - self._current_size_bytes)
            if self._max_encoder_tokens is not None:
                remaining_tokens = max(
                    0, self._max_encoder_tokens - self._current_encoder_tokens
                )
            else:
                remaining_tokens = None

            freeable_bytes = sum(
                self._cache[h].data_size_bytes
                for h in self._freeable_hashes
                if h in self._cache
            )
            freeable_tokens = sum(
                self._cache[h].num_tokens
                for h in self._freeable_hashes
                if h in self._cache
            )

            if self._max_entry_count is not None:
                remaining_entries = max(
                    0, self._max_entry_count - len(self._cache)
                )
            else:
                remaining_entries = None

            return EncoderBudget(
                remaining_bytes=remaining_bytes,
                remaining_tokens=remaining_tokens,
                remaining_entries=remaining_entries,
                freeable_bytes=freeable_bytes,
                freeable_tokens=freeable_tokens,
                total_bytes=self._max_size_bytes,
                used_bytes=self._current_size_bytes,
                total_tokens=self._max_encoder_tokens,
                used_tokens=self._current_encoder_tokens,
            )

    @property
    def current_size_bytes(self) -> int:
        return self._current_size_bytes

    @property
    def num_entries(self) -> int:
        return len(self._cache)

    @property
    def num_active(self) -> int:
        return len(self._active_hashes)

    @property
    def num_freeable(self) -> int:
        return len(self._freeable_hashes)

    @property
    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, int]:
        """Return a snapshot of cache statistics."""
        with self._lock:
            return {
                "num_entries": len(self._cache),
                "num_active": len(self._active_hashes),
                "num_freeable": len(self._freeable_hashes),
                "current_size_bytes": self._current_size_bytes,
                "current_encoder_tokens": self._current_encoder_tokens,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
            }

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, mm_hash: int) -> bool:
        return self.has(mm_hash)

    # ------------------------------------------------------------------
    # Internal helpers (must be called with self._lock held)
    # ------------------------------------------------------------------

    def _put_locked(
        self,
        mm_hash: int,
        data: torch.Tensor,
        num_tokens: int,
    ) -> bool:
        data_size = _get_tensor_size(data)

        # Update existing entry
        existing = self._cache.get(mm_hash)
        if existing is not None and existing.state != CacheEntryState.FREED:
            old_size = existing.data_size_bytes
            old_tokens = existing.num_tokens
            existing.data = data
            existing.num_tokens = num_tokens
            existing.data_size_bytes = data_size
            existing.last_access_time = time.monotonic()
            self._current_size_bytes += data_size - old_size
            self._current_encoder_tokens += num_tokens - old_tokens
            self._cache.move_to_end(mm_hash)
            return True

        # Evict freeable entries if needed to make room
        needed = data_size - max(0, self._max_size_bytes - self._current_size_bytes)
        if needed > 0:
            freed = self._evict_freeable_locked(needed)
            if self._current_size_bytes + data_size > self._max_size_bytes:
                # Still not enough space after evicting everything freeable
                return False

        # Check entry count limit
        if (
            self._max_entry_count is not None
            and len(self._cache) >= self._max_entry_count
        ):
            # Try to evict one freeable entry
            if not self._freeable_hashes:
                return False
            oldest = self._get_lru_freeable_locked()
            if oldest is not None:
                self._evict_entry_locked(oldest)
            else:
                return False

        # Check token budget
        if (
            self._max_encoder_tokens is not None
            and self._current_encoder_tokens + num_tokens > self._max_encoder_tokens
        ):
            # Try freeing until we have room
            tokens_needed = (
                self._current_encoder_tokens
                + num_tokens
                - self._max_encoder_tokens
            )
            self._evict_freeable_by_tokens_locked(tokens_needed)
            if self._current_encoder_tokens + num_tokens > self._max_encoder_tokens:
                return False

        # Insert new entry
        entry = EncoderCacheEntry(
            mm_hash=mm_hash,
            data=data,
            num_tokens=num_tokens,
            ref_count=0,
            state=CacheEntryState.FREEABLE,
        )
        self._cache[mm_hash] = entry
        self._freeable_hashes.add(mm_hash)
        self._current_size_bytes += data_size
        self._current_encoder_tokens += num_tokens
        return True

    def _evict_entry_locked(self, mm_hash: int) -> bool:
        entry = self._cache.get(mm_hash)
        if entry is None:
            return False
        if entry.state == CacheEntryState.ACTIVE:
            # Cannot evict an entry with active references
            return False
        if entry.state == CacheEntryState.FREED:
            return False
        # Perform eviction
        self._current_size_bytes -= entry.data_size_bytes
        self._current_encoder_tokens -= entry.num_tokens
        self._freeable_hashes.discard(mm_hash)
        self._active_hashes.discard(mm_hash)
        del self._cache[mm_hash]
        return True

    def _evict_freeable_locked(self, target_bytes: int) -> int:
        """Evict LRU freeable entries until target_bytes freed or none left."""
        freed = 0
        # Collect freeable in LRU order (iteration order of OrderedDict)
        to_evict = []
        for h, entry in self._cache.items():
            if entry.state == CacheEntryState.FREEABLE:
                to_evict.append(h)
                freed += entry.data_size_bytes
                if target_bytes > 0 and freed >= target_bytes:
                    break

        actual_freed = 0
        for h in to_evict:
            entry = self._cache.get(h)
            if entry is not None:
                actual_freed += entry.data_size_bytes
                self._current_size_bytes -= entry.data_size_bytes
                self._current_encoder_tokens -= entry.num_tokens
                self._freeable_hashes.discard(h)
                del self._cache[h]
        return actual_freed

    def _evict_freeable_by_tokens_locked(self, target_tokens: int) -> int:
        """Evict LRU freeable entries until target_tokens freed."""
        freed_tokens = 0
        to_evict = []
        for h, entry in self._cache.items():
            if entry.state == CacheEntryState.FREEABLE:
                to_evict.append(h)
                freed_tokens += entry.num_tokens
                if freed_tokens >= target_tokens:
                    break

        actual_freed = 0
        for h in to_evict:
            entry = self._cache.get(h)
            if entry is not None:
                actual_freed += entry.num_tokens
                self._current_size_bytes -= entry.data_size_bytes
                self._current_encoder_tokens -= entry.num_tokens
                self._freeable_hashes.discard(h)
                del self._cache[h]
        return actual_freed

    def _get_lru_freeable_locked(self) -> Optional[int]:
        """Return the hash of the least recently used freeable entry."""
        for h, entry in self._cache.items():
            if entry.state == CacheEntryState.FREEABLE:
                return h
        return None


@dataclass
class EncoderBudget:
    """Snapshot of remaining encoder cache capacity.

    Returned by ``EncoderCacheManager.compute_budget()`` so the scheduler
    can decide whether to admit new encoder work.
    """

    remaining_bytes: int
    remaining_tokens: Optional[int]
    remaining_entries: Optional[int]
    freeable_bytes: int
    freeable_tokens: int
    total_bytes: int
    used_bytes: int
    total_tokens: Optional[int]
    used_tokens: int

    @property
    def available_bytes(self) -> int:
        """Total bytes that could be used (remaining + freeable)."""
        return self.remaining_bytes + self.freeable_bytes

    @property
    def available_tokens(self) -> Optional[int]:
        """Total tokens that could be used (remaining + freeable)."""
        if self.remaining_tokens is None:
            return None
        return self.remaining_tokens + self.freeable_tokens

    def can_admit(self, size_bytes: int, num_tokens: int = 0) -> bool:
        """Check if a new entry of the given size could be admitted
        (possibly after evicting freeable entries)."""
        if self.available_bytes < size_bytes:
            return False
        if self.remaining_tokens is not None:
            avail_tokens = self.available_tokens
            if avail_tokens is not None and avail_tokens < num_tokens:
                return False
        return True


class EncoderCacheType(Enum):
    """Namespace keys for the two-tier encoder cache."""

    FEATURE = "mm_feature"
    EMBEDDING = "mm_embedding"


class MultiTierEncoderCacheManager:
    """Convenience wrapper that manages separate ``EncoderCacheManager``
    instances for raw encoder features and projected embeddings.

    This allows the scheduler to cache ViT outputs (features) separately
    from the final projected embeddings, enabling partial re-use: for
    example, if only the projection layer changes across requests, the
    raw features can still be reused.

    Parameters:
        feature_cache_bytes: Max bytes for the feature cache.
        embedding_cache_bytes: Max bytes for the embedding cache.
        max_encoder_tokens: Optional global token cap applied to each tier.
    """

    def __init__(
        self,
        feature_cache_bytes: int,
        embedding_cache_bytes: int,
        max_encoder_tokens: Optional[int] = None,
    ):
        self._caches: Dict[EncoderCacheType, EncoderCacheManager] = {
            EncoderCacheType.FEATURE: EncoderCacheManager(
                max_size_bytes=feature_cache_bytes,
                max_encoder_tokens=max_encoder_tokens,
            ),
            EncoderCacheType.EMBEDDING: EncoderCacheManager(
                max_size_bytes=embedding_cache_bytes,
                max_encoder_tokens=max_encoder_tokens,
            ),
        }

    def get_cache(self, tier: EncoderCacheType) -> EncoderCacheManager:
        """Return the cache for a specific tier."""
        return self._caches[tier]

    @property
    def feature_cache(self) -> EncoderCacheManager:
        return self._caches[EncoderCacheType.FEATURE]

    @property
    def embedding_cache(self) -> EncoderCacheManager:
        return self._caches[EncoderCacheType.EMBEDDING]

    def clear(self) -> None:
        for cache in self._caches.values():
            cache.clear()

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        return {
            tier.value: cache.get_stats()
            for tier, cache in self._caches.items()
        }
