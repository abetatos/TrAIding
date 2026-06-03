"""
Shared I/O utilities: parquet read/write, HTTP with caching and retry logic.
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# ── Parquet helpers ──────────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)


def load_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(path, **kwargs)


# ── HTTP cache ───────────────────────────────────────────────────────────────

class HttpCache:
    """Disk cache for GET requests. Keyed by URL + params hash."""

    def __init__(self, cache_dir: str | Path = ".cache/http"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, url: str, params: dict) -> str:
        raw = url + json.dumps(params, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, url: str, params: dict) -> Any | None:
        path = self.cache_dir / f"{self._key(url, params)}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def set(self, url: str, params: dict, data: Any) -> None:
        path = self.cache_dir / f"{self._key(url, params)}.json"
        path.write_text(json.dumps(data))


# ── Rate-limited session ─────────────────────────────────────────────────────

class ThrottledSession:
    """
    Wraps requests.Session with per-call rate limiting and exponential backoff.
    min_interval_s: minimum seconds between requests (e.g. 0.2 = 5 req/s).
    """

    def __init__(
        self,
        min_interval_s: float = 0.25,
        cache: HttpCache | None = None,
    ):
        self.session = requests.Session()
        self.min_interval_s = min_interval_s
        self.cache = cache
        self._last_call = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_call = time.monotonic()

    @retry(
        retry=retry_if_exception_type(requests.HTTPError),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    def get(self, url: str, params: dict | None = None, **kwargs) -> dict | list:
        params = params or {}

        if self.cache:
            cached = self.cache.get(url, params)
            if cached is not None:
                return cached

        self._throttle()
        resp = self.session.get(url, params=params, timeout=30, **kwargs)

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 10))
            time.sleep(retry_after)
            resp.raise_for_status()

        resp.raise_for_status()
        data = resp.json()

        if self.cache:
            self.cache.set(url, params, data)

        return data
