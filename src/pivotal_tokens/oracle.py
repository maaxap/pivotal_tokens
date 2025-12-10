import logging
import re
import typing as t
from abc import ABC, abstractmethod


DEFAULT_ORACLE_RESPONSE_REGEX = r"(?s)\A.*</think>(?!.*</think>)(.*)\Z"
# DEFAULT_ORACLE_ANSWER_REGEX = r"(?s)<\|im_start\|>assistant.*?(?:</think>\s*)?Answer:\s*(.*?)\s*(?=(?:<\|im_end\|>|<\|endoftext\|>|\Z))"
DEFAULT_ORACLE_ANSWER_REGEX = r"(?s)\s*(?:Answer:\s*)?(.*?)\s*(?=(?:<\|im_end\|>|<\|endoftext\|>|\Z))"


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


class RegexOracle(Oracle):

    def __init__(self,
                 answer_regex: str = DEFAULT_ORACLE_ANSWER_REGEX,
                 response_regex: str = DEFAULT_ORACLE_RESPONSE_REGEX,
                 fuzzy_match_threshold: float | None = None):
        self.answer_regex = answer_regex
        self.response_regex = response_regex
        self.fuzzy_match_threshold = fuzzy_match_threshold

    def verify(self, actual: t.Any, expected: t.Sequence[t.Any]) -> bool:
        extracted_response = self.extract_text(actual, self.response_regex)
        if extracted_response is None:
            logging.debug(f"Oracle verification failed, no response extracted from completion: {actual}.")
            return False
    
        extracted_answer = self.extract_text(extracted_response, self.answer_regex)
        if extracted_answer is None:
            logging.debug(f"Oracle verification failed, no answer extracted from completion: {actual}.")
            return False

        result = False
        if self.fuzzy_match_threshold is not None:
            similarities = [compute_similarity(extracted_answer, expected_answer)
                            for expected_answer in expected]
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            logging.debug(f"Avg. similarity {avg_similarity} for extracted_answer='{extracted_answer}' "
                          f"and expected='{expected}'")
    
            result = any(sim >= self.fuzzy_match_threshold for sim in similarities)
        else:
            normalized_extracted = normalize_text(extracted_answer)
            result = any(normalized_extracted == normalize_text(expected_anser)
                         for expected_anser in expected)
            
        match_type = "fuzzy" if self.fuzzy_match_threshold is not None else "exact"
        logging.debug(f"Result of oracle verification with {match_type} match: "
                    #   f"extracted_response='{extracted_response}', "
                      f"extracted_answer='{extracted_answer}', expected='{expected}', "
                      f"result={result}")

        return result

    def extract_text(self, completion: str, regex: str) -> str | None:
        match = re.search(regex, completion, re.IGNORECASE | re.DOTALL)

        try:
            answer = match.group(1).strip()

        # If the regex does not match, it will raise an AttributeError
        # and we return None to indicate no answer was found.
        except AttributeError:
            answer = None
        
        return answer
