import argparse
import json
import logging
import typing as t
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

from pivotal_tokens.extractor import THINKING_END_TOKEN, LogLikelihoodSpikeExtractor
from pivotal_tokens.hf.dataset import load_hotpotqa_dataset
from pivotal_tokens.hf.loading import load_model, load_tokenizer
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

    mode: t.Literal["expected_answer", "actual_answer"] = field(metadata={"help": "Answer mode."})

    explicit_truncation: bool = field(metadata={"help": "Whether to finish reasoning trace with <truncated> to explicitly indicate truncation."}, default=False)

    debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode."})


def load_config(path: Path) -> Config:
    config_dict = json.loads(path.read_text())
    return Config(**config_dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract log-likelihood tokens from traces using a JSON config file.")
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

    sample_id_to_trace = traces_df.set_index("id")["trace"].to_dict()
    samples = [s for s in samples if s.id in sample_id_to_trace]
    logging.info(f"Filtered to {len(samples)} samples with available traces.")

    model = load_model(model_id=config.model_id, device=config.device)
    tokenizer = load_tokenizer(model_id=config.model_id)
    model.eval()
    logging.info(f"Loaded model and tokenizer from {config.model_id}.")

    base_repo = DictRepo(dirpath=Path(config.repo_dirpath))
    extractor = LogLikelihoodSpikeExtractor(model=model,
                                            tokenizer=tokenizer,
                                            mode=config.mode,
                                            base_repo=base_repo,
                                            explicit_truncation=config.explicit_truncation)

    accumulator = []
    for idx, sample in enumerate(samples, start=1):
        logging.info(f"Processing sample {sample.id} ({idx}/{len(samples)})...")
        trace = sample_id_to_trace.get(sample.id)

        if trace.endswith(THINKING_END_TOKEN):
            trace = trace[:-len(THINKING_END_TOKEN)]

        try:
            pivotal_tokens = extractor.extract(sample_id=sample.id,
                                               reasoning_trace=trace,
                                               system_prompt=config.system_prompt,
                                               user_prompt=sample.query,
                                               actual_answer=sample.ground_truth,
                                               expected_answer=sample.ground_truth,
                                               metadata=sample.metadata)
            for token in pivotal_tokens:
                accumulator.append(asdict(token))

        except Exception as exc:
            logging.error(f"Extraction failed for sample {sample.id}: {exc}", exc_info=True, stack_info=True)
            raise exc

    output_df = pd.DataFrame(accumulator)
    output_df.to_csv(output_filepath, index=False)
    logging.info(f"Saved extracted pivotal tokens to {output_filepath}")


if __name__ == "__main__":
    parsed_args = parse_args()
    config = load_config(parsed_args.config)

    level = logging.DEBUG if config.debug else logging.INFO
    setup_logging(level=level)

    main(config=config)
