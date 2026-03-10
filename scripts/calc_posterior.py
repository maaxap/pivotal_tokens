import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import nltk.tokenize
import pandas as pd
from transformers import GenerationConfig

from pivotal_tokens.extractor import SuccessProbabilityShiftExtractor, THINKING_END_TOKEN, THINKING_START_TOKEN
from pivotal_tokens.hf.dataset import Sample, load_hotpotqa_dataset
from pivotal_tokens.hf.loading import load_model, load_tokenizer
from pivotal_tokens.oracle import RegexOracle
from pivotal_tokens.repo import NoopRepo
from pivotal_tokens.utils import setup_logging


@dataclass
class Config:
    traces_filepath: str = field(metadata={"help": "Path to traces CSV with model thinking traces."})
    output_filepath: str = field(metadata={"help": "Path to output CSV with posterior probabilities."})

    model_id: str = field(metadata={"help": "Hugging Face model id to load."})
    device: str = field(metadata={"help": "Device string to force (e.g., 'cuda', 'cpu')."})
    system_prompt: str = field(metadata={"help": "System prompt text used for generation."})

    max_new_tokens: int = field(metadata={"help": "Max new tokens per completion."})
    temperature: float = field(metadata={"help": "Sampling temperature."})
    top_p: float = field(metadata={"help": "Top-p nucleus sampling parameter."})
    top_k: int = field(metadata={"help": "Top-k sampling parameter."})
    min_p: float = field(metadata={"help": "Minimum probability cutoff for sampling."})
    do_sample: bool = field(metadata={"help": "Enable sampling (otherwise greedy decoding)."})

    num_trials: int = field(metadata={"help": "Trials per probability estimate."})
    batch_size: int = field(metadata={"help": "Batch size for generation in probability estimation."})
    fuzzy_match_threshold: float = field(metadata={"help": "Jaccard similarity threshold for oracle fuzzy match."})

    dataset_split: str | None = field(default=None,
                                      metadata={"help": "Optional HotpotQA split fallback when CSV misses fields."})
    dataset_name: str | None = field(default=None,
                                     metadata={"help": "Optional HotpotQA name fallback when CSV misses fields."})

    debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode."})


def load_config(path: Path) -> Config:
    config_dict = json.loads(path.read_text())
    return Config(**config_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate posterior success probabilities for each trace.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON file containing configuration.")
    return parser.parse_args()


def build_sample_lookup(config: Config, need_lookup: bool) -> dict[str, Sample]:
    if not need_lookup:
        return {}

    if config.dataset_split is None or config.dataset_name is None:
        raise ValueError("Missing query/ground_truth columns in traces CSV. "
                         "Provide dataset_split and dataset_name in config for fallback lookup.")

    samples = load_hotpotqa_dataset(split=config.dataset_split, name=config.dataset_name)
    return {sample.id: sample for sample in samples}


def normalize_trace(trace: str) -> tuple[str, bool, bool]:
    normalized = trace.strip()
    if normalized == "":
        raise ValueError("Trace is empty after stripping")

    trace_had_thinking_start = normalized.startswith(THINKING_START_TOKEN)
    if not trace_had_thinking_start:
        normalized = f"{THINKING_START_TOKEN}{normalized}"

    trace_was_already_closed = THINKING_END_TOKEN in normalized

    # if trace_was_already_closed:
    #     normalized = normalized.split(THINKING_END_TOKEN)[0]

    if not trace_was_already_closed:
        normalized = f"{normalized}\n{THINKING_END_TOKEN}" 

    return normalized, trace_was_already_closed, trace_had_thinking_start


def main(config: Config):
    output_filepath = Path(config.output_filepath)
    if output_filepath.exists():
        raise FileExistsError(f"Output file {output_filepath} already exists, skipping extraction.")

    traces_df = pd.read_csv(config.traces_filepath)
    logging.info(f"Loaded {len(traces_df)} traces from {config.traces_filepath}.")

    required_cols = {"id", "trace", "query", "ground_truth"}
    missing_cols = required_cols - set(traces_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in traces CSV: {sorted(missing_cols)}")

    model = load_model(model_id=config.model_id, device=config.device)
    tokenizer = load_tokenizer(model_id=config.model_id)
    model.eval()
    logging.info(f"Loaded model and tokenizer from {config.model_id}.")

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
    noop_repo = NoopRepo()
    extractor = SuccessProbabilityShiftExtractor(model=model,
                                                 tokenizer=tokenizer,
                                                 oracle=oracle,
                                                 base_repo=noop_repo,
                                                 prob_threshold=0,
                                                 num_trials=config.num_trials,
                                                 min_prob=0,
                                                 max_prob=1,
                                                 batch_size=config.batch_size,
                                                 generation_config=generation_config)

    records = []
    num_skipped = 0
    for idx, row in traces_df.iterrows():
        sample_id = str(row["id"])
        logging.info(f"Processing sample {sample_id} ({idx + 1}/{len(traces_df)})")

        query = row["query"]
        ground_truth = row["ground_truth"]

        trace = row["trace"]
        trace_normalized, _, _ = normalize_trace(trace=trace)
    
        extractor.clear_cache()
        prob_post = extractor.estimate_success_probability(prefix=trace_normalized,
                                                            system_prompt=config.system_prompt,
                                                            user_prompt=str(query),
                                                            expected_answer=str(ground_truth),
                                                            sample_repo=noop_repo,
                                                            metadata={})

        record = row.to_dict()
        record["query"] = query
        record["ground_truth"] = ground_truth
        record["prob_post"] = prob_post
        record["trace_normalized"] = trace_normalized
        records.append(record)


    output_df = pd.DataFrame(records)
    output_df.to_csv(output_filepath, index=False)
    logging.info(f"Saved posterior results for {len(output_df)} samples to {output_filepath}.")
    logging.info(f"Processed={len(output_df)}, skipped={num_skipped}.")


if __name__ == "__main__":
    parsed_args = parse_args()
    config = load_config(parsed_args.config)

    level = logging.DEBUG if config.debug else logging.INFO
    setup_logging(level=level)

    main(config=config)
