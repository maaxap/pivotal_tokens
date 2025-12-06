import logging
from dataclasses import dataclass
from typing import Any, Literal

from datasets import load_dataset

from pivotal_tokens.constants import get_hf_cache_dir


@dataclass
class Sample:
    id: str
    query: str
    ground_truth: str
    metadata: dict[str, Any]

    def __repr__(self):
        return f"Sample(id={self.id}, query={self.query}, ground_truth={self.ground_truth}"


# TODO: double check that 'fullwiki' is the intended configuration
def load_hotpotqa_dataset(split: Literal["train", "validation", "test"], name: str = "fullwiki") -> list[Sample]:
    """Load the HotpotQA dataset from Hugging Face Datasets.
    :seealso: https://huggingface.co/datasets/hotpotqa/hotpot_qa
    
    :param split: The dataset split to load ('train', 'validation', or 'test').
    :return: A list of Sample objects containing the dataset samples.
    """
    raw_dataset = load_dataset("hotpotqa/hotpot_qa", name=name, split=split, cache_dir=get_hf_cache_dir())
    
    samples = []
    for i in range(len(raw_dataset)):
        raw_sample = raw_dataset[i]
        sample = Sample(id=raw_sample["id"],
                        query=raw_sample["question"],
                        ground_truth=raw_sample["answer"],
                        metadata=dict(raw_sample))
        samples.append(sample)

    logging.debug(f"Loaded {len(samples)} samples from HotpotQA {split} split")
    return samples


def load_socrates_dataset() -> list[Sample]:
    # https://huggingface.co/datasets/soheeyang/SOCRATES
    pass
