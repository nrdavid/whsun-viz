"""Per-image reads out of the sharded matrix_assets ZIP store.

The 3,540 loose .webp are packed into 59 ZIP_STORED shards plus a manifest. STORE, not DEFLATE:
webp is already compressed, and byte-verbatim entries let git delta successive shard versions.

GLIQUID_MATRIX_SHARDS overrides the shard directory -- the seam for serving from a mounted
volume instead of the image.
"""
from __future__ import annotations

import os
import threading
import zipfile
from pathlib import Path

# Not `immutable`: asset urls are stable across rebuilds, so it would pin returning visitors to
# a stale figure for a year.
CACHE_CONTROL = "public, max-age=604800"
MANIFEST_NAME = "manifest.tsv"
SHARD_PREFIX = "assets_"


class MatrixAssetStore:
    def __init__(self, shards_dir: Path):
        self._dir = Path(shards_dir)
        self._entries: dict[str, tuple[str, str]] = {}   # name -> (shard, etag)
        self._zips: dict[str, zipfile.ZipFile] = {}
        self._locks: dict[str, threading.Lock] = {}

    @classmethod
    def load(cls, shards_dir=None) -> "MatrixAssetStore | None":
        """Return a ready store, or None if no shard set is present.

        None is not an error -- it lets a tree that still carries loose matrix_assets/ fall
        through to the StaticFiles mount.
        """
        shards_dir = Path(os.environ.get("GLIQUID_MATRIX_SHARDS") or shards_dir)
        store = cls(shards_dir)
        manifest = store._dir / MANIFEST_NAME
        if not manifest.is_file():
            return None

        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            name, shard, dig, _size = line.split("\t")
            store._entries[name] = (shard, f'"{dig}"')
        if not store._entries:
            return None

        # Open every shard up front so a missing or truncated one fails at startup, not on a
        # visitor's first hover.
        for shard in sorted({s for s, _ in store._entries.values()}):
            path = store._dir / f"{SHARD_PREFIX}{shard}.zip"
            store._zips[shard] = zipfile.ZipFile(path)
            store._locks[shard] = threading.Lock()
        return store

    def __len__(self) -> int:
        return len(self._entries)

    def etag(self, name: str) -> str | None:
        entry = self._entries.get(name)
        return entry[1] if entry else None

    def read(self, name: str) -> bytes | None:
        entry = self._entries.get(name)
        if entry is None:
            # The manifest is the allowlist, so an unknown name never reaches the filesystem.
            return None
        shard, _ = entry
        # ZipFile.open bumps _fileRefCnt unguarded and the docs promise nothing about concurrent
        # use; the lock costs microseconds against a page-cache hit.
        with self._locks[shard]:
            return self._zips[shard].read(name)
