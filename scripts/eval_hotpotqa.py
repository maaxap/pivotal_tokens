"""Generate model completions for HotpotQA, extract thinking traces, and score with an oracle."""

import argparse
import hashlib
import logging
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import GenerationConfig, set_seed

from pivotal_tokens.hf.dataset import load_hotpotqa_dataset
from pivotal_tokens.hf.loading import load_model, load_tokenizer
from pivotal_tokens.hf.sampling import batch_sampling, extract_thinking_trace
from pivotal_tokens.hf.dataset import Sample
from pivotal_tokens.oracle import RegexOracle
from pivotal_tokens.repo import DictRepo
from pivotal_tokens.utils import setup_logging


# Model / hardware
MODEL_ID = "Qwen/Qwen3-1.7B"
DEVICE = "cpu"

# Dataset
DATASET_SPLIT = "validation"
DATASET_NAME = "fullwiki"

# Prompting
SYSTEM_PROMPT = ("Answer the following question directly, strictly without chain-of-thought after \"</think>\"."
                 "Keep the answer short (e.g., \"yes\" or \"no\" for binary questions, a person's full name if "
                 "the question expects a person, a country name if it asks about a country, etc.). Output the "
                 "answer strictly after the prefix \"Answer: \"  with no extra text.")

ENABLE_THINKING = True

# Generation
BATCH_SIZE = 16
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0.0
DO_SAMPLE = True

# Oracle
FUZZY_MATCH_THRESHOLD = 0.7

# Output
OUTPUT_PATH = Path("experiments/exp0.1_eval/data/hotpotqa_fullwiki_qwen3_1.7B_results.csv")
CACHE_DIR = Path("experiments/exp0.1_eval/cache")

# Run control
RANDOM_SEED = 42


@dataclass
class Config:
    # Model / hardware
    model_id: str = field(default=MODEL_ID, metadata={"help": "Hugging Face model id to load."})
    device: str = field(default=DEVICE, metadata={"help": "Device string to force (e.g., 'cuda', 'cpu')."})

    # Dataset
    dataset_split: str = field(default=DATASET_SPLIT, metadata={"help": "HotpotQA split (train/validation/test)."})
    dataset_name: str = field(default=DATASET_NAME, metadata={"help": "HotpotQA configuration name (e.g., fullwiki)."})

    # Prompting
    system_prompt: str = field(default=SYSTEM_PROMPT, metadata={"help": "System prompt text used for generation."})
    enable_thinking: bool = field(default=ENABLE_THINKING, metadata={"help": "Use thinking tokens in the chat template."})

    # Generation
    batch_size: int = field(default=BATCH_SIZE, metadata={"help": "Samples to generate per batch."})
    max_new_tokens: int = field(default=MAX_NEW_TOKENS, metadata={"help": "Max new tokens per completion."})
    temperature: float = field(default=TEMPERATURE, metadata={"help": "Sampling temperature."})
    top_p: float = field(default=TOP_P, metadata={"help": "Top-p nucleus sampling parameter."})
    top_k: int = field(default=TOP_K, metadata={"help": "Top-k sampling parameter."})
    min_p: float = field(default=MIN_P, metadata={"help": "Minimum probability cutoff for sampling."})
    do_sample: bool = field(default=DO_SAMPLE, metadata={"help": "Enable sampling (otherwise greedy decoding)."})

    # Oracle
    fuzzy_match_threshold: float = field(default=FUZZY_MATCH_THRESHOLD, metadata={"help": "Jaccard similarity threshold for oracle fuzzy match."})
    
    # Output
    output_path: str = field(default=str(OUTPUT_PATH), metadata={"help": "CSV path for results."})

    # Run control
    seed: int = field(default=RANDOM_SEED, metadata={"help": "Random seed for reproducibility."})
    debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode."})


def load_config(path: Path) -> Config:
    config_dict = json.loads(path.read_text())
    return Config(**config_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HotpotQA completions based on a JSON config file.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON file containing EvalArgs configuration.")
    return parser.parse_args() 


def build_completion_cache_key(sample_query: str, system_prompt: str, generation_config: GenerationConfig) -> str:
    payload = {
        "system_prompt": system_prompt,
        "user_prompt": sample_query,
        "generation_config": generation_config.to_dict(),
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]


def build_cache_payload(sample: Sample, completion: str, trace: str | None, oracle_result: bool) -> dict[str, Any]:
    return {
        **asdict(sample),
        "completion": completion,
        "oracle_result": oracle_result,
        "trace": trace,
    }


def main(config: Config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    set_seed(config.seed)

    device = config.device 
    model = load_model(model_id=config.model_id, device=device)
    tokenizer = load_tokenizer(model_id=config.model_id)
    model.eval()

    generation_config = GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        min_p=config.min_p,
        do_sample=config.do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )

    oracle = RegexOracle(fuzzy_match_threshold=config.fuzzy_match_threshold)
    repo = DictRepo(dirpath=CACHE_DIR)
    samples = load_hotpotqa_dataset(split=config.dataset_split, name=config.dataset_name)
    cache_path = f"{config.dataset_name}/{config.dataset_split}"

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    unprocessed = []
    for sample in samples:
        cache_key = build_completion_cache_key(sample_query=sample.query,
                                               system_prompt=config.system_prompt,
                                               generation_config=generation_config)
        result = repo.load(path=cache_path, key=cache_key)
        if result is None:
            unprocessed.append(sample)
            continue

        results.append(result)
            

    for i in range(0, len(unprocessed), config.batch_size):
        samples = unprocessed[i:i + config.batch_size]

        try:
            completions = batch_sampling(samples=samples,
                                         system_prompt=config.system_prompt,
                                         model=model,
                                         tokenizer=tokenizer,
                                         generation_config=generation_config,
                                         enable_thinking=config.enable_thinking)
        except Exception as e:
            logging.error(f"Batch generation failed with error: {e}")
            continue

        for sample, completion in zip(samples, completions):
            try:
                trace = None
                if config.enable_thinking:
                    trace = extract_thinking_trace(completion)

                oracle_result = oracle.verify(actual=completion, expected=[sample.ground_truth])

                payload = build_cache_payload(sample=sample,
                                              completion=completion,
                                              trace=trace,
                                              oracle_result=oracle_result)
                cache_key = build_completion_cache_key(sample_query=sample.query,
                                                       system_prompt=config.system_prompt,
                                                       generation_config=generation_config)
                repo.save(path=cache_path, key=cache_key, data=payload)
                results.append(payload)
            except Exception as e:
                logging.error(f"Postprocessing failed for sample {sample.id}: {e}")

    df = pd.DataFrame(results)
    df["metadata"] = df["metadata"].apply(lambda x: json.dumps(x, indent=2))
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parsed_args = parse_args()
    config = load_config(parsed_args.config)

    level = logging.DEBUG if config.debug else logging.INFO
    setup_logging(level=level)

    main(config=config)
