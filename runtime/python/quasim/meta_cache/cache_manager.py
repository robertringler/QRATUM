"""Cache manager for compiled meta-kernels with versioning."""
from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class CacheEntry:
    """Represents a cached compiled kernel."""
    kernel_hash: str
    backend: str
    version: str
    compilation_time: float
    source_code: str
    metadata: dict[str, str]
    
    def to_json(self) -> str:
        """Serialize cache entry to JSON."""
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_json(cls, data: str) -> CacheEntry:
        """Deserialize cache entry from JSON."""
        obj = json.loads(data)
        return cls(**obj)


class CacheManager:
    """Manages compiled kernel cache with versioning."""
    
    def __init__(self, cache_dir: Optional[pathlib.Path] = None) -> None:
        if cache_dir is None:
            cache_dir = pathlib.Path.home() / ".quasim" / "meta_cache"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._version = "1.0.0"
        
    def compute_hash(self, source: str, backend: str) -> str:
        """Compute hash for kernel source and backend."""
        content = f"{backend}:{source}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:16]
        
    def get(self, kernel_hash: str) -> Optional[CacheEntry]:
        """Retrieve cached kernel by hash."""
        cache_file = self.cache_dir / f"{kernel_hash}.json"
        if not cache_file.exists():
            return None
            
        try:
            data = cache_file.read_text()
            return CacheEntry.from_json(data)
        except (json.JSONDecodeError, TypeError, KeyError):
            return None
            
    def put(self, entry: CacheEntry) -> None:
        """Store compiled kernel in cache."""
        cache_file = self.cache_dir / f"{entry.kernel_hash}.json"
        cache_file.write_text(entry.to_json())
        
    def invalidate(self, kernel_hash: str) -> bool:
        """Remove kernel from cache."""
        cache_file = self.cache_dir / f"{kernel_hash}.json"
        if cache_file.exists():
            cache_file.unlink()
            return True
        return False
        
    def clear(self) -> int:
        """Clear all cached kernels. Returns number of entries removed."""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count
        
    def list_entries(self) -> list[CacheEntry]:
        """List all cached kernel entries."""
        entries = []
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                data = cache_file.read_text()
                entries.append(CacheEntry.from_json(data))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
        return entries
