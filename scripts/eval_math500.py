"""Generate model completions for GSM8K, extract thinking traces, and score with an oracle."""

import argparse
import hashlib
import logging
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import GenerationConfig, set_seed

from pivotal_tokens.hf.dataset import load_math500_dataset, Sample
from pivotal_tokens.hf.loading import load_model, load_tokenizer
from pivotal_tokens.hf.generation import generate_batch, extract_thinking_trace
from pivotal_tokens.oracle import RegexOracle
from pivotal_tokens.repo import DictRepo
from pivotal_tokens.utils import setup_logging


@dataclass
class Config:
    # Output
    output_path: str = field(metadata={"help": "CSV path for results."})
    cache_dir: str = field(metadata={"help": "Directory path for completion cache."})

    # Model / hardware
    model_id: str = field(metadata={"help": "Hugging Face model id to load."})
    device: str = field(metadata={"help": "Device string to force (e.g., 'cuda', 'cpu')."})

    # Prompting
    system_prompt: str = field(metadata={"help": "System prompt text used for generation."})
    enable_thinking: bool = field(metadata={"help": "Use thinking tokens in the chat template."})

    # Generation
    batch_size: int = field(metadata={"help": "Samples to generate per batch."})
    max_new_tokens: int = field(metadata={"help": "Max new tokens per completion."})
    temperature: float = field(metadata={"help": "Sampling temperature."})
    top_p: float = field(metadata={"help": "Top-p nucleus sampling parameter."})
    top_k: int = field(metadata={"help": "Top-k sampling parameter."})
    min_p: float = field(metadata={"help": "Minimum probability cutoff for sampling."})
    do_sample: bool = field(metadata={"help": "Enable sampling (otherwise greedy decoding)."})

    # Oracle
    fuzzy_match_threshold: float = field(metadata={"help": "Jaccard similarity threshold for oracle fuzzy match."})

    # Run control
    seed: int = field(metadata={"help": "Random seed for reproducibility."})
    debug: bool = field(metadata={"help": "Whether to run in debug mode."})


def load_config(path: Path) -> Config:
    config_dict = json.loads(path.read_text())
    return Config(**config_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GSM8K completions based on a JSON config file.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON config file.")
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
    set_seed(config.seed)

    tokenizer = load_tokenizer(model_id=config.model_id)
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
    repo = DictRepo(dirpath=Path(config.cache_dir))
    samples = load_math500_dataset()
    cache_path = "math500"

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    unprocessed: list[tuple[Sample, str]] = []
    for sample in samples:
        cache_key = build_completion_cache_key(
            sample_query=sample.query,
            system_prompt=config.system_prompt,
            generation_config=generation_config,
        )
        try:
            result = repo.load(path=cache_path, key=cache_key)
            results.append(result)
        except FileNotFoundError as e:
            unprocessed.append((sample, cache_key))

    logging.info(f"Found {len(results)} cached completions, {len(unprocessed)} to process.")

    if unprocessed:
        model = load_model(model_id=config.model_id, device=config.device)
        model.eval()

        for i in range(0, len(unprocessed), config.batch_size):
            batch = unprocessed[i:i + config.batch_size]
            batch_samples = [s for s, _ in batch]

            try:
                completions = generate_batch(
                    samples=batch_samples,
                    system_prompt=config.system_prompt,
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    enable_thinking=config.enable_thinking,
                )
            except Exception as e:
                logging.error(f"Batch generation failed: {e}")
                continue

            for (sample, cache_key), completion in zip(batch, completions):
                try:
                    trace = extract_thinking_trace(completion) if config.enable_thinking else None
                    oracle_result = oracle.verify(actual=completion, expected=[sample.ground_truth])
                    payload = build_cache_payload(
                        sample=sample,
                        completion=completion,
                        trace=trace,
                        oracle_result=oracle_result,
                    )
                    repo.save(path=cache_path, key=cache_key, data=payload)
                    results.append(payload)
                except Exception as e:
                    logging.error(f"Postprocessing failed for sample {sample.id}: {e}")

    df = pd.DataFrame(results)
    df["metadata"] = df["metadata"].apply(lambda x: json.dumps(x, indent=2))
    df.to_csv(output_path, index=False)
    logging.info(f"Saved results to {output_path}.")


if __name__ == "__main__":
    parsed_args = parse_args()
    config = load_config(parsed_args.config)
    setup_logging(logging.DEBUG if config.debug else logging.INFO)
    main(config=config)
