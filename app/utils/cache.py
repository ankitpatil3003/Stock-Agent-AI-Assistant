import os
import io
import time
import pickle
import hashlib
from typing import Any, Optional


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class FileCache:
    """
    Super-simple file-based cache.
    - Stores pickled blobs under CACHE_DIR
    - TTL (seconds) is enforced on read
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _path_for(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{_sha1(key)}.pkl")

    def get(self, key: str, ttl_sec: int) -> Optional[Any]:
        path = self._path_for(key)
        if not os.path.exists(path):
            return None
        try:
            # Expiry check
            mtime = os.path.getmtime(path)
            if (time.time() - mtime) > ttl_sec:
                return None
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path_for(key)
        tmp = path + ".tmp"
        try:
            with open(tmp, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except Exception:
            # Best-effort cache; ignore failures
            pass
