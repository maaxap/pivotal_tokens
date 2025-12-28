import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd
from transformers import GenerationConfig

from pivotal_tokens.extractor import SuccessProbabilityShiftExtractor
from pivotal_tokens.hf.dataset import load_hotpotqa_dataset
from pivotal_tokens.hf.loading import load_model, load_tokenizer
from pivotal_tokens.oracle import RegexOracle
from pivotal_tokens.repo import DictRepo
from pivotal_tokens.utils import setup_logging


@dataclass
class Config:
    traces_filepath: str = field(metadata={"help": "Path to traces CSV with model thinking traces."})
    output_filepath: str = field(metadata={"help": "Path to output CSV with identified pivotal tokens."})
    repo_dirpath: str = field(metadata={"help": "Path to repository directory for storing intermediate data."})

    model_id: str = field(metadata={"help": "Hugging Face model id to load."})
    device: str = field(metadata={"help": "Device string to force (e.g., 'cuda', 'cpu')."})

    dataset_split: str = field(metadata={"help": "HotpotQA split (train/validation/test)."})
    dataset_name: str = field(metadata={"help": "HotpotQA configuration name (e.g., fullwiki)."})
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
    parser = argparse.ArgumentParser(description="Extract SPS pivotal tokens from traces using a JSON config file.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON file containing configuration.")
    return parser.parse_args()


def main(config: Config):
    output_filepath = Path(config.output_filepath)
    if output_filepath.exists():
        raise FileExistsError(f"Output file {output_filepath} already exists, skipping extraction.")

    samples = load_hotpotqa_dataset(split=config.dataset_split, name=config.dataset_name)
    logging.info(f"Loaded {len(samples)} samples from HotpotQA {config.dataset_split} split.")

    traces_df = pd.read_csv(config.traces_filepath)
    logging.info(f"Loaded {len(traces_df)} traces from {config.traces_filepath}.")

    sample_id_to_trace = traces_df.set_index('id')['trace'].to_dict()
    samples = [s for s in samples if s.id in sample_id_to_trace]
    logging.info(f"Filtered to {len(samples)} samples with available traces.")

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
    base_repo = DictRepo(dirpath=Path(config.repo_dirpath))
    extractor = SuccessProbabilityShiftExtractor(model=model,
                                                 tokenizer=tokenizer,
                                                 oracle=oracle,
                                                 base_repo=base_repo,
                                                 prob_threshold=config.prob_threshold,
                                                 num_trials=config.num_trials,
                                                 min_prob=config.min_prob,
                                                 max_prob=config.max_prob,
                                                 batch_size=config.batch_size,
                                                 generation_config=generation_config)

    n_processed_samples = 0
    accumulator = []
    for sample in samples:
        logging.info(f"Processing sample {sample.id} ({n_processed_samples + 1}/{len(samples)})...")
        trace = sample_id_to_trace.get(sample.id, None)
        if trace is None:
            logging.warning(f"No trace found for sample {sample.id}, skipping.")
            continue

        extractor.clear_cache()
        try:
            pivotal_tokens = extractor.extract(sample_id=sample.id,
                                               reasoning_trace=trace,
                                               system_prompt=config.system_prompt,
                                               user_prompt=sample.query,
                                               actual_answer="",
                                               expected_answer=sample.ground_truth,
                                               metadata=sample.metadata)
            for token in pivotal_tokens:
                accumulator.append(asdict(token))

        except Exception as exc:
            logging.error(f"Extraction failed for sample {sample.id}: {exc}")

        n_processed_samples += 1

    output_df = pd.DataFrame(accumulator)
    output_df.to_csv(output_filepath, index=False)
    logging.info(f"Saved extracted pivotal tokens to {output_filepath}.")


if __name__ == "__main__":
    parsed_args = parse_args()
    config = load_config(parsed_args.config)

    level = logging.DEBUG if config.debug else logging.INFO
    setup_logging(level=level)

    main(config=config)
