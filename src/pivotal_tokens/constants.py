from pathlib import Path


def get_project_root_dir() -> Path:
    project_root = Path(__file__).parent.parent.parent
    return project_root.resolve()


def get_data_dir() -> Path:
    data_dir = get_project_root_dir() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_hf_cache_dir() -> Path:
    hf_cache_dir = get_project_root_dir() / ".cache" / "huggingface"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    return hf_cache_dir
