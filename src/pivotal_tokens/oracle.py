import typing as t
from abc import ABC, abstractmethod


class Oracle(ABC):
    @abstractmethod
    def verify(self, actual: t.Any, expected: t.Sequence[t.Any]) -> bool:
        pass

    @abstractmethod
    def extract_answer(self, completion: str) -> t.Any:
        pass


class RegexOracle(Oracle):
    def __init__(self, answer_regex: str):
        self.answer_regex = answer_regex

    def verify(self, actual: t.Any, expected: t.Sequence[t.Any]) -> bool:
        pass

    def extract_answer(self, completion: str) -> t.Any:
        pass
