import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from transformers import GenerationConfig

from pivotal_tokens.extractor import SuccessProbabilityShiftExtractor
from pivotal_tokens.hf.generation import generate_next_token
from pivotal_tokens.hf.loading import load_model, load_tokenizer
from pivotal_tokens.oracle import RegexOracle
from pivotal_tokens.repo import NoopRepo, SampleRepo
from pivotal_tokens.utils import setup_logging


@dataclass
class Config:
    pivotal_tokens_filepath: str = field(metadata={"help": "Path to pivotal tokens CSV."})
    output_filepath: str = field(metadata={"help": "Path to output CSV with alternatives."})

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

    debug: bool = field(metadata={"help": "Whether to run in debug mode."})


def load_config(path: Path) -> Config:
    config_dict = json.loads(path.read_text())
    return Config(**config_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify token alternatives from pivotal token spans.")
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


def build_prefix(extractor: SuccessProbabilityShiftExtractor,
                 system_prompt: str,
                 user_prompt: str,
                 pivotal_context: str) -> str | None:
    context = extractor.create_context(system_prompt=system_prompt, user_prompt=user_prompt)
    if pivotal_context.startswith(context):
        prefix = pivotal_context[len(context):]

        logging.debug(f"Derived prefix: {prefix!r}")

        return prefix

    idx = pivotal_context.find(context)
    if idx != -1:
        logging.warning("Pivotal context mismatch; using substring match to derive prefix.")
        return pivotal_context[idx + len(context):]

    logging.error("Unable to derive prefix from pivotal context; skipping row.")
    return None


def main(config: Config):
    output_filepath = Path(config.output_filepath)
    if output_filepath.exists():
        raise FileExistsError(f"Output file {output_filepath} already exists, skipping extraction.")

    df = pd.read_csv(config.pivotal_tokens_filepath)
    df['token_ids'] = df['token_ids'].apply(json.loads)
    logging.info(f"Loaded {len(df)} pivotal spans from {config.pivotal_tokens_filepath}.")

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

    records = []
    for row in df.itertuples(index=False):
        user_prompt = row.query
        expected_answer = row.ground_truth
        token_ids = row.token_ids

        prefix = build_prefix(extractor=extractor,
                              system_prompt=config.system_prompt,
                              user_prompt=user_prompt,
                              pivotal_context=row.pivotal_context)

        if prefix is None:
            continue

        next_step = generate_next_token(context=row.pivotal_context, model=model, tokenizer=tokenizer)

        token_logprob_orig = None
        token_id = token_ids[0]

        for _, token_id, token_logprob in next_step.vocab_tuples:
            if token_id == token_ids[0]:
                token_logprob_orig = float(token_logprob)
                break

        alternatives = select_alternatives(next_step.vocab_tuples)
        if not alternatives:
            logging.warning(f"No alternative tokens found for span {row.span_id}")
            continue

        for alt in alternatives:
            alt_token_id = int(alt["token_id"])
            alt_token_logprob = float(alt["token_logprob"])
            alt_text = tokenizer.decode([alt_token_id], skip_special_tokens=False)
            alt_prefix = prefix + alt_text

            prob_after_alt = extractor.estimate_success_probability(prefix=alt_prefix,
                                                                    system_prompt=config.system_prompt,
                                                                    user_prompt=user_prompt,
                                                                    expected_answer=expected_answer,
                                                                    sample_repo=NoopRepo(),
                                                                    metadata={},)

            records.append({"sample_id": row.sample_id,
                            "span_id": row.span_id,
                            "token_ids": row.token_ids,
                            "span_text": row.span_text,
                            "prob_before": row.prob_before,
                            "prob_after": row.prob_after,
                            "prob_delta": row.prob_delta,
                            "is_pivotal": row.is_pivotal,
                            "pivotal_context": row.pivotal_context,
                            "metadata": row.metadata,
                            "prob_init": row.prob_init,
                            "span_text_alt": alt_text,
                            "token_ids_alt": alt_token_id,
                            "token_logprob_alt": alt_token_logprob,
                            "token_logprob_orig": token_logprob_orig,
                            "prob_after_alt": prob_after_alt,
                            "alt_type": alt["alt_type"]})

    output_df = pd.DataFrame(records)
    output_df.to_csv(output_filepath, index=False)
    logging.info(f"Saved {len(output_df)} alternative token rows to {output_filepath}.")


if __name__ == "__main__":
    parsed_args = parse_args()
    config = load_config(parsed_args.config)

    level = logging.DEBUG if config.debug else logging.INFO
    setup_logging(level=level)

    main(config=config)
