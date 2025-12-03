import logging
import re

from pathlib import Path


def setup_logger(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    

def get_project_root_dir() -> Path:
    project_root = Path(__file__).parent.parent
    return project_root.resolve()


def get_data_dir() -> Path:
    data_dir = get_project_root_dir() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_hf_cache_dir() -> Path:
    hf_cache_dir = get_project_root_dir() / ".cache" / "huggingface"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    return hf_cache_dir


def normalize_text(text: str) -> str:
    if not text:
        return ""

    normalized = text.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized


def compute_similarity(text1: str, text2: str) -> float:
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    
    if not words1 or not words2:
        return 0.0
        
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0