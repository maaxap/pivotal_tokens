from ast import Not
import json
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path


class Repo(ABC):
    @abstractmethod
    def save(self, path: str, key: str, data: dict[str, t.Any]) -> None:
        """
        Save dict to disk.

        :param path: Relative path from repo root (e.g., "trials", "sample_001/subdivisions")
        :param key: Unique identifier for this data item (without .json extension)
        :param data: Data to save
        """
        pass

    @abstractmethod
    def load(self, path: str, key: str) -> dict[str, t.Any]:
        """
        Load dict from disk.

        :param path: Relative path from repo root
        :param key: Unique identifier for the data item (without .json extension)

        :returns: Loaded data dictionary
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list[str]:
        """
        List all keys in the given path.

        :param path: Relative path from repo root

        :returns: List of keys (without .json extension)
        """
        pass


class NoopRepo(Repo):
    def save(self, path: str, key: str, data: dict[str, t.Any]) -> None:
        pass

    def load(self, path: str, key: str) -> dict[str, t.Any]:
        raise NotImplementedError("NoopRepo does not support loading data")
    
    def list(self, path: str) -> list[str]:
        raise NotImplementedError("NoopRepo does not support listing data")



class DictRepo(Repo):
    def __init__(self, dirpath: Path) -> None:
        self.dirpath = dirpath

    def save(self, path: str, key: str, data: dict[str, t.Any]) -> None:
        filepath = self.dirpath / path / f"{key}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.is_file():
            raise KeyError(f"File {filepath} already exists")

        filepath.write_text(json.dumps(data, indent=2))

    def load(self, path: str, key: str) -> dict[str, t.Any]:
        filepath = self.dirpath / path / f"{key}.json"
        if not filepath.is_file():
            raise FileNotFoundError(f"File {filepath} does not exist")

        return json.loads(filepath.read_text())
    
    def list(self, path: str) -> list[str]:
        dirpath = self.dirpath / path
        if not dirpath.is_dir():
            return []
        
        return [f.stem for f in dirpath.glob("*.json")]
    

class SampleRepo(Repo):
    """
    Simple wrapper around base Repo that prefixes all paths with sample_id.
    """

    def __init__(self, sample_id: str, base_repo: Repo):
        self.sample_id = sample_id
        self.base_repo = base_repo

    def save(self, path: str, key: str, data: dict[str, t.Any]) -> None:
        """Save with sample_id prefix: {sample_id}/{path}"""
        prefixed_path = f"{self.sample_id}/{path}" if path else self.sample_id
        self.base_repo.save(path=prefixed_path, key=key, data=data)

    def load(self, path: str, key: str) -> dict[str, t.Any]:
        """Load with sample_id prefix: {sample_id}/{path}"""
        prefixed_path = f"{self.sample_id}/{path}" if path else self.sample_id
        return self.base_repo.load(path=prefixed_path, key=key)
    
    def list(self, path: str) -> list[str]:
        """List with sample_id prefix: {sample_id}/{path}"""
        prefixed_path = f"{self.sample_id}/{path}" if path else self.sample_id
        return self.base_repo.list(path=prefixed_path)

