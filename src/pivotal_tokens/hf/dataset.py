import logging
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
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



def load_gsm8k_dataset(split: Literal["train", "test"], name: str = "main") -> list[Sample]:
    """Load the GSM8K dataset from Hugging Face Datasets.
    :seealso: https://huggingface.co/datasets/openai/gsm8k
    
    :param split: The dataset split to load ('train' or 'test').
    :param name: The dataset subset ('main' or 'socratic').
    :return: A list of Sample objects containing the dataset samples.
    """
    raw_dataset = load_dataset("openai/gsm8k", name=name, split=split, cache_dir=get_hf_cache_dir())

    samples = []
    for i in range(len(raw_dataset)):
        raw_sample = raw_dataset[i]
        answer_full = raw_sample["answer"]
        answer_final = answer_full.split("####")[-1].strip()
        sample = Sample(id=str(i),
                        query=raw_sample["question"],
                        ground_truth=answer_final,
                        metadata=dict(raw_sample))
        samples.append(sample)

    logging.debug(f"Loaded {len(samples)} samples from GSM8K {split} split")
    return samples


def load_math500_dataset() -> list[Sample]:
    """Load the MATH-500 dataset from Hugging Face Datasets.
    :seealso: https://huggingface.co/datasets/HuggingFaceH4/MATH-500

    :return: A list of Sample objects containing the dataset samples.
    """
    raw_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=get_hf_cache_dir())

    samples = []
    for i in range(len(raw_dataset)):
        raw_sample = raw_dataset[i]
        sample = Sample(id=raw_sample["unique_id"],
                        query=raw_sample["problem"],
                        ground_truth=raw_sample["answer"],
                        metadata=dict(raw_sample))
        samples.append(sample)

    logging.debug(f"Loaded {len(samples)} samples from MATH-500 test split")
    return samples


def load_imo_answer_bench_dataset() -> list[Sample]:
    """Load the IMOAnswerBench dataset from GitHub.
    :seealso: https://github.com/google-deepmind/superhuman

    :return: A list of Sample objects containing the dataset samples.
    """
    url = "https://raw.githubusercontent.com/google-deepmind/superhuman/refs/heads/main/imobench/answerbench_v2.csv"
    df = pd.read_csv(url)

    samples = []
    for i, row in df.iterrows():
        sample = Sample(id=row["Problem ID"],
                        query=row["Problem"],
                        ground_truth=row["Short Answer"],
                        metadata=row.to_dict())
        samples.append(sample)

    logging.debug(f"Loaded {len(samples)} samples from IMOAnswerBench dataset")
    return samples