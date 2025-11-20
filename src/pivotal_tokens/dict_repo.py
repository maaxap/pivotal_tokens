import json
import logging
import typing as t
from pathlib import Path


class DictRepo:
    def __init__(self, dirpath: Path) -> None:
        self.dirpath = dirpath

    def _make_filename(self, key: str) -> str:
        return f"{key}.json"

    def save(self, key: str, data: dict[str, t.Any], name: str, overwrite: bool = False) -> None:
        """Save dict to disk with given key"""
        filepath = self.dirpath / name / self._make_filename(key)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.is_file():
            logging.warning(f"File {filepath} already exists")

            if not overwrite:
                return

        dumped = json.dumps(data)
        filepath.write_text(dumped)

    def load(self, key: str, name: str) -> dict:
        """Load dict from disk by key"""
        filepath = self.dirpath / name / self._make_filename(key)
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} does not exist")

        return json.loads(filepath.read_text())
