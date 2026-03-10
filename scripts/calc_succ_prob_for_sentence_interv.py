import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from transformers import GenerationConfig

from pivotal_tokens.extractor import (
    THINKING_END_TOKEN,
    THINKING_START_TOKEN,
    SuccessProbabilityShiftExtractor,
)
from pivotal_tokens.hf.loading import load_model, load_tokenizer
from pivotal_tokens.oracle import RegexOracle
from pivotal_tokens.repo import NoopRepo
from pivotal_tokens.utils import setup_logging


@dataclass
class Config:
    interventions_filepath: str = field(metadata={"help": "Path to interventions CSV."})
    output_filepath: str = field(metadata={"help": "Path to output CSV with intervention probabilities."})

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

    debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode."})


def load_config(path: Path) -> Config:
    config_dict = json.loads(path.read_text())
    return Config(**config_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate intervention success probabilities for sentence spans.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON file containing configuration.")
    return parser.parse_args()


def normalize_trace(trace: str) -> str:
    normalized = str(trace).strip()
    if normalized == "":
        raise ValueError("Trace is empty after stripping")

    if not normalized.startswith(THINKING_START_TOKEN):
        normalized = f"{THINKING_START_TOKEN}\n{normalized}"

    if THINKING_END_TOKEN not in normalized:
        normalized = f"{normalized}\n{THINKING_END_TOKEN}"

    return normalized


def build_prefix(extractor: SuccessProbabilityShiftExtractor,
                 system_prompt: str,
                 user_prompt: str,
                 pivotal_context: str) -> str | None:
    context = extractor.create_context(system_prompt=system_prompt, user_prompt=user_prompt)
    pivotal_context = str(pivotal_context)

    if pivotal_context.startswith(context):
        return pivotal_context[len(context):]

    logging.error("Unable to derive prefix from pivotal context; skipping row.")
    return None


def main(config: Config):
    output_filepath = Path(config.output_filepath)
    if output_filepath.exists():
        raise FileExistsError(f"Output file {output_filepath} already exists, skipping extraction.")

    interventions_df = pd.read_csv(config.interventions_filepath, keep_default_na=False)
    logging.info(f"Loaded {len(interventions_df)} intervention rows from {config.interventions_filepath}.")

    required_cols = {
        "intervention_id",
        "sample_id",
        "span_id",
        "query",
        "ground_truth",
        "pivotal_context_interv",
        "span_text_interv",
        "trace_interv",
        "intervention_type",
    }
    missing_cols = required_cols - set(interventions_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in interventions CSV: {sorted(missing_cols)}")

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
    for idx, row in interventions_df.iterrows():
        intervention_id = str(row.get("intervention_id", f"row_{idx}"))
        logging.info(f"Processing intervention {intervention_id} ({idx + 1}/{len(interventions_df)})")

        query = str(row["query"])
        ground_truth = str(row["ground_truth"])
        pivotal_context_original = str(row["pivotal_context"])
        pivotal_context_interv = str(row["pivotal_context_interv"])
        span_text_original = str(row["span_text"])
        span_text_interv = str(row["span_text_interv"])

        prefix_before_original = build_prefix(extractor=extractor,
                                            system_prompt=config.system_prompt,
                                            user_prompt=query,
                                            pivotal_context=pivotal_context_original)
        
        if prefix_before_original is None:
            raise ValueError(f"Got None prefix before original for the row: {row}")

        prefix_before_interv = build_prefix(extractor=extractor,
                                            system_prompt=config.system_prompt,
                                            user_prompt=query,
                                            pivotal_context=pivotal_context_interv)
        if prefix_before_interv is None:
            raise ValueError(f"Got None prefix before interv for the row: {row}")

        try:
            trace_original_normalized = normalize_trace(trace=str(row["trace_original"]))
            trace_interv_normalized = normalize_trace(trace=str(row["trace_interv"]))
        except ValueError as exc:
            raise ValueError(f"Failed to normalize traces for the row: {row}") from exc
        
        prefix_after_original = prefix_before_original + span_text_original
        prefix_after_interv = prefix_before_interv + span_text_interv

        prob_before_original = extractor.estimate_success_probability(prefix=prefix_before_original,
                                                                    system_prompt=config.system_prompt,
                                                                    user_prompt=query,
                                                                    expected_answer=ground_truth,
                                                                    sample_repo=noop_repo,
                                                                    metadata={
                                                                        "stage": "prob_before_original",
                                                                        "intervention_id": intervention_id,
                                                                    })
        prob_after_original = extractor.estimate_success_probability(prefix=prefix_after_original,
                                                                   system_prompt=config.system_prompt,
                                                                   user_prompt=query,
                                                                   expected_answer=ground_truth,
                                                                   sample_repo=noop_repo,
                                                                   metadata={
                                                                       "stage": "prob_after_original",
                                                                       "intervention_id": intervention_id,
                                                                   })
        prob_post_original = extractor.estimate_success_probability(prefix=trace_original_normalized,
                                                                  system_prompt=config.system_prompt,
                                                                  user_prompt=query,
                                                                  expected_answer=ground_truth,
                                                                  sample_repo=noop_repo,
                                                                  metadata={
                                                                      "stage": "prob_post_original",
                                                                      "intervention_id": intervention_id,
                                                                  })
        

        prob_before_interv = extractor.estimate_success_probability(prefix=prefix_before_interv,
                                                                    system_prompt=config.system_prompt,
                                                                    user_prompt=query,
                                                                    expected_answer=ground_truth,
                                                                    sample_repo=noop_repo,
                                                                    metadata={
                                                                        "stage": "prob_before_interv",
                                                                        "intervention_id": intervention_id,
                                                                    })
        prob_after_interv = extractor.estimate_success_probability(prefix=prefix_after_interv,
                                                                   system_prompt=config.system_prompt,
                                                                   user_prompt=query,
                                                                   expected_answer=ground_truth,
                                                                   sample_repo=noop_repo,
                                                                   metadata={
                                                                       "stage": "prob_after_interv",
                                                                       "intervention_id": intervention_id,
                                                                   })
        prob_post_interv = extractor.estimate_success_probability(prefix=trace_interv_normalized,
                                                                  system_prompt=config.system_prompt,
                                                                  user_prompt=query,
                                                                  expected_answer=ground_truth,
                                                                  sample_repo=noop_repo,
                                                                  metadata={
                                                                      "stage": "prob_post_interv",
                                                                      "intervention_id": intervention_id,
                                                                  })

        record = row.to_dict()
        record["prob_before_original"] = prob_before_original
        record["prob_after_original"] = prob_after_original
        record["prob_delta_original"] = prob_after_original - prob_before_original
        record["prob_post_original"] = prob_post_original

        record["prob_before_interv"] = prob_before_interv
        record["prob_after_interv"] = prob_after_interv
        record["prob_delta_interv"] = prob_after_interv - prob_before_interv
        record["prob_post_interv"] = prob_post_interv

        record["trace_original_normalized"] = trace_original_normalized
        record["trace_interv_normalized"] = trace_interv_normalized
        records.append(record)

    output_df = pd.DataFrame(records)
    output_df.to_csv(output_filepath, index=False)
    logging.info(f"Saved {len(output_df)} intervention rows to {output_filepath}.")
    logging.info(f"Processed={len(output_df)}, skipped={num_skipped}.")


if __name__ == "__main__":
    parsed_args = parse_args()
    config = load_config(parsed_args.config)

    level = logging.DEBUG if config.debug else logging.INFO
    setup_logging(level=level)

    main(config=config)
