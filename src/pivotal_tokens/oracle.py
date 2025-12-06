import re
import typing as t
from abc import ABC, abstractmethod


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


class Oracle(ABC):
    @abstractmethod
    def verify(self, actual: t.Any, expected: t.Sequence[t.Any]) -> bool:
        pass

    @abstractmethod
    def extract_answer(self, completion: str) -> t.Any:
        pass


class RegexOracle(Oracle):
    def __init__(self, answer_regex: str, fuzzy_match_threshold: float | None = None):
        self.answer_regex = answer_regex
        self.fuzzy_match_threshold = fuzzy_match_threshold

    def verify(self, actual: t.Any, expected: t.Sequence[t.Any]) -> bool:
        extracted_answer = self.extract_answer(actual)
        if extracted_answer is None:
            return False

        result = False
        if self.fuzzy_match_threshold is not None:
            similarity = any(compute_similarity(extracted_answer, expected_answer)
                             for expected_answer in expected)
            result = similarity >= self.fuzzy_match_threshold
        else:
            normalized_extracted = normalize_text(extracted_answer)
            result = any(normalized_extracted == normalize_text(expected_anser)
                         for expected_anser in expected)

        return result

    def extract_answer(self, completion: str) -> str | None:
        match = re.search(self.answer_regex, completion, re.IGNORECASE | re.DOTALL)

        try:
            answer = match.group(1).strip()

        # If the regex does not match, it will raise an AttributeError
        # and we return None to indicate no answer was found.
        except AttributeError:
            answer = None
        
        return answer
