import argparse
import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from transformers import GenerationConfig, set_seed

from pivotal_tokens.extractor import SuccessProbabilityShiftExtractor, THINKING_END_TOKEN, THINKING_START_TOKEN
from pivotal_tokens.hf.generation import generate_next_token
from pivotal_tokens.hf.loading import load_model, load_tokenizer
from pivotal_tokens.oracle import RegexOracle
from pivotal_tokens.repo import DictRepo, NoopRepo
from pivotal_tokens.utils import setup_logging


@dataclass
class Config:
    traces_filepath: str = field(metadata={"help": "Path to traces CSV with model thinking traces."})
    output_filepath: str = field(metadata={"help": "Path to output CSV with random token alternatives."})
    repo_dirpath: str = field(metadata={"help": "Path to repo directory."})

    model_id: str = field(metadata={"help": "Hugging Face model id to load."})
    device: str = field(metadata={"help": "Device string to force (e.g., 'cuda', 'cpu')."})

    system_prompt: str = field(metadata={"help": "System prompt text used for generation."})

    max_new_tokens: int = field(metadata={"help": "Max new tokens per completion."})
    temperature: float = field(metadata={"help": "Sampling temperature."})
    top_p: float = field(metadata={"help": "Top-p nucleus sampling parameter."})
    top_k: int = field(metadata={"help": "Top-k sampling parameter."})
    min_p: float = field(metadata={"help": "Minimum probability cutoff for sampling."})
    do_sample: bool = field(metadata={"help": "Enable sampling (otherwise greedy decoding)."})

    prob_threshold: float = field(metadata={"help": "Minimum probability delta for pivotal span."})
    num_trials: int = field(metadata={"help": "Trials per probability estimate."})
    min_prob: float = field(metadata={"help": "Min probability threshold to proceed with extraction."})
    max_prob: float = field(metadata={"help": "Max probability threshold to proceed with extraction."})
    batch_size: int = field(metadata={"help": "Batch size for generation in probability estimation."})

    fuzzy_match_threshold: float = field(metadata={"help": "Jaccard similarity threshold for oracle fuzzy match."})

    seed: int = field(metadata={"help": "Random seed for reproducibility."})
    debug: bool = field(metadata={"help": "Whether to run in debug mode."})


def load_config(path: Path) -> Config:
    config_dict = json.loads(path.read_text())
    return Config(**config_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample random tokens from traces and test alternatives.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON file containing configuration.")
    return parser.parse_args()


def select_alternatives(vocab_tuples: list[tuple[str, int, float]]) -> list[dict[str, float | int | str]]:
    entries = []
    for token_str, token_id, token_logprob in vocab_tuples:
        if not math.isfinite(token_logprob):
            continue
        entries.append({"token_str": token_str,
                        "token_id": token_id,
                        "token_logprob": token_logprob})

    if not entries:
        return []

    entries = sorted(entries, key=lambda x: x["token_logprob"], reverse=True)

    logging.debug(f"Top alternative tokens: {entries[:5]}")
    logging.debug(f"Last alternative tokens: {entries[-5:]}")

    picks = []
    num_top_entries = min(3, len(entries))
    for i in range(num_top_entries):
        entry = entries[i]
        picks.append(dict(**entry, alt_type=f"top_{i + 1}"))

    if len(entries) >= 3:
        median_idx = len(entries) // 2
        median_entry = entries[median_idx]
        picks.append(dict(**median_entry, alt_type="median"))

        last_idx = len(entries) - 1
        last_entry = entries[last_idx]
        picks.append(dict(**last_entry, alt_type="lowest"))

    return picks


def select_random_token_indices(token_count: int,
                                rng: random.Random) -> list[tuple[int, str, int, int]]:
    one_third = token_count // 3
    two_third = (2 * token_count) // 3
    ranges = [
        ("0-33", 0, one_third),
        ("33-66", one_third, two_third),
        ("66-100", two_third, token_count),
    ]

    picks = []
    for label, start, end in ranges:
        if start >= end:
            return []
        picks.append((rng.randrange(start, end), label, start, end))

    return picks


def main(config: Config):
    output_filepath = Path(config.output_filepath)
    if output_filepath.exists():
        raise FileExistsError(f"Output file {output_filepath} already exists, skipping extraction.")

    traces_path = Path(config.traces_filepath)

    df = pd.read_csv(traces_path)
    logging.info(f"Loaded {len(df)} traces from {traces_path}.")

    model = load_model(model_id=config.model_id, device=config.device)
    tokenizer = load_tokenizer(model_id=config.model_id)
    logging.info(f"Loaded model and tokenizer from {config.model_id}.")

    model.eval()

    generation_config = GenerationConfig(max_new_tokens=config.max_new_tokens,
                                         temperature=config.temperature,
                                         top_p=config.top_p,
                                         top_k=config.top_k,
                                         min_p=config.min_p,
                                         do_sample=config.do_sample,
                                         pad_token_id=tokenizer.pad_token_id)

    oracle = RegexOracle(fuzzy_match_threshold=config.fuzzy_match_threshold)
    extractor = SuccessProbabilityShiftExtractor(model=model,
                                                 tokenizer=tokenizer,
                                                 oracle=oracle,
                                                 base_repo=NoopRepo(),
                                                 prob_threshold=config.prob_threshold,
                                                 num_trials=config.num_trials,
                                                 min_prob=config.min_prob,
                                                 max_prob=config.max_prob,
                                                 batch_size=config.batch_size,
                                                 generation_config=generation_config)

    cache_repo = DictRepo(dirpath=Path(config.repo_dirpath))
    cache_path = "rand_tokens"

    rng = random.Random(config.seed)
    set_seed(config.seed)

    thinking_token_ids = {tokenizer.convert_tokens_to_ids(THINKING_START_TOKEN),
                          tokenizer.convert_tokens_to_ids(THINKING_END_TOKEN)}

    records = []
    noop_repo = NoopRepo()
    for idx, row in df.iterrows():
        logging.info(f"Processing row {idx}/{len(df)}")
        sample_id = row.id
        user_prompt = row.query
        expected_answer = row.ground_truth
        trace = row.trace

        extractor.clear_cache()

        token_ids = tokenizer.encode(trace, add_special_tokens=False)
        token_ids = [tid for tid in token_ids if tid not in thinking_token_ids]
        
        token_count = len(token_ids)
        if token_count < 3:
            logging.warning(f"Trace too short for sample {sample_id} ({token_count} tokens), skipping.")
            continue

        picks = select_random_token_indices(token_count=token_count, rng=rng)

        logging.debug(f"Selected random token indices for sample {sample_id}: {picks}")

        if not picks:
            raise Exception(f"Unable to select random token ranges for sample {sample_id}")

        context = extractor.create_context(system_prompt=config.system_prompt,
                                           user_prompt=user_prompt)

        for token_index, range_label, range_start, range_end in picks:
            prefix_ids = token_ids[:token_index]
            token_id = token_ids[token_index]

            prefix = tokenizer.decode(prefix_ids, skip_special_tokens=False)
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)

            logging.debug(f"Sample {sample_id}, token index {token_index} ({range_label}): "
                          f"token_id={token_id}, token_text='{token_text}'")
            logging.debug(f"Prefix: '{prefix}'")

            pivotal_context = context + prefix

            next_step = generate_next_token(context=pivotal_context, model=model, tokenizer=tokenizer)

            token_logprob_random = None
            for _, cand_id, cand_logprob in next_step.vocab_tuples:
                if cand_id == token_id:
                    token_logprob_random = float(cand_logprob)
                    break

            alternatives = select_alternatives(next_step.vocab_tuples)
            if not alternatives:
                raise Exception(f"No alternative tokens found for sample {sample_id}")

            prob_before = extractor.estimate_success_probability(prefix=prefix,
                                                                system_prompt=config.system_prompt,
                                                                user_prompt=user_prompt,
                                                                expected_answer=expected_answer,
                                                                sample_repo=noop_repo,
                                                                metadata={})
            prob_after = extractor.estimate_success_probability(prefix=prefix + token_text,
                                                                system_prompt=config.system_prompt,
                                                                user_prompt=user_prompt,
                                                                expected_answer=expected_answer,
                                                                sample_repo=noop_repo,
                                                                metadata={})
            prob_delta = prob_after - prob_before

            span_payload = f"{sample_id}:{token_index}:{range_label}"
            span_id = hashlib.sha256(span_payload.encode("utf-8")).hexdigest()

            for alt in alternatives:
                alt_token_id = int(alt["token_id"])
                cache_payload = f"{sample_id}:{token_index}:{range_label}:{alt_token_id}"
                cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()
                try:
                    record = cache_repo.load(path=cache_path, key=cache_key)
                    records.append(record)
                    logging.debug(f"Cache hit for sample {sample_id}, token index {token_index}, "
                                  f"alt token id {alt_token_id}, skipping computation.")
                    continue
                except FileNotFoundError:
                    pass

                alt_token_logprob = float(alt["token_logprob"])
                alt_text = tokenizer.decode([alt_token_id], skip_special_tokens=False)
                alt_prefix = prefix + alt_text

                prob_after_alt = extractor.estimate_success_probability(prefix=alt_prefix,
                                                                        system_prompt=config.system_prompt,
                                                                        user_prompt=user_prompt,
                                                                        expected_answer=expected_answer,
                                                                        sample_repo=noop_repo,
                                                                        metadata={})

                record = {"sample_id": sample_id,
                          "query": user_prompt,
                          "ground_truth": expected_answer,
                          "span_id": span_id,
                          "token_ids_random": [token_id],
                          "span_text_random": token_text,
                          "token_index_random": token_index,
                          "token_range_label": range_label,
                          "token_range_start": range_start,
                          "token_range_end": range_end,
                          "trace_token_count": token_count,
                          "prob_before": prob_before,
                          "prob_after": prob_after,
                          "prob_delta": prob_delta,
                          "pivotal_context": pivotal_context,
                          "span_text_alt": alt_text,
                          "token_ids_alt": [alt_token_id],
                          "token_logprob_alt": alt_token_logprob,
                          "token_logprob_random": token_logprob_random,
                          "prob_after_alt": prob_after_alt,
                          "alt_type": alt["alt_type"]}
                cache_repo.save(path=cache_path, key=cache_key, data=record)
                records.append(record)

    output_df = pd.DataFrame(records)
    output_df.to_csv(output_filepath, index=False)
    logging.info(f"Saved {len(output_df)} random token rows to {output_filepath}.")


if __name__ == "__main__":
    parsed_args = parse_args()
    config = load_config(parsed_args.config)

    level = logging.DEBUG if config.debug else logging.INFO
    setup_logging(level=level)

    main(config=config)
