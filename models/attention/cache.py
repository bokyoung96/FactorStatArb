from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch


class ResidualCache:
    def __init__(self, cache_dir: Path, enable_disk: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_disk = enable_disk
        self._memory: Dict[str, torch.Tensor] = {}
        self._meta_path = self.cache_dir / "meta.json"

    def _safe_key(self, date_key: str) -> str:
        """
        Normalize to filename-safe string. If parsable as date, use YYYYMMDD;
        otherwise fall back to stripping unsafe chars.
        """
        try:
            ts = pd.to_datetime(date_key)
            return ts.strftime("%Y%m%d")
        except Exception:
            return re.sub(r"[^0-9A-Za-z_-]", "-", str(date_key))

    def _year_dir(self, date_key: str) -> Path:
        try:
            ts = pd.to_datetime(date_key)
            year = ts.year
        except Exception:
            year = "unknown"
        year_dir = self.cache_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        return year_dir

    def _path(self, date_key: str) -> Path:
        key = self._safe_key(date_key)
        return self._year_dir(date_key) / f"{key}.pt"

    def get(self, date_key: str) -> Optional[torch.Tensor]:
        key = self._safe_key(date_key)
        if key in self._memory:
            return self._memory[key]
        if not self.enable_disk:
            return None
        path = self._path(key)
        if path.exists():
            tensor = torch.load(path, map_location="cpu")
            self._memory[key] = tensor
            return tensor
        return None

    def set(self, date_key: str, tensor: torch.Tensor) -> None:
        key = self._safe_key(date_key)
        self._memory[key] = tensor.detach().cpu()
        if self.enable_disk:
            path = self._path(key)
            torch.save(self._memory[key], path)
            logging.info("Residual cache saved date=%s key=%s path=%s", date_key, key, path)

    def save_meta(self, meta: dict) -> None:
        if not self.enable_disk:
            return
        with self._meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
