import logging
import random
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import numpy as np
import torch  # TODO: Missing import - torch.manual_seed is called below
from sympy.physics.units import temperature
from transformers import GenerationConfig, set_seed

from pivotal_tokens.hf.loading import load_model, load_tokenizer
from pivotal_tokens.extractor import SuccessProbabilityShiftExtractor
from pivotal_tokens.oracle import RegexOracle
from pivotal_tokens.repo import DictRepo
from pivotal_tokens.utils import setup_logging


# TODO: Consider using transformers.set_seed() which sets all seeds at once
# TODO: Seed should be configurable via CLI argument for multiple experimental runs
RANDOM_SEED = 42

# TODO: Consider making configurable via CLI arguments or config file
HF_MODEL_ID = "Qwen/Qwen3-1.7B"
DICT_REPO_DIR = Path("./repo")

# TODO: Inconsistent instruction - says "without chain-of-thought after </think>" but also expects </think> to exist
# TODO: Consider separating this into separate prompts for CoT and non-CoT modes
# TODO: The phrase "strictly without chain-of-thought after </think>" is confusing - clarify the intent
SYSTEM_PROMPT = ("Answer the following question directly, strictly without chain-of-thought after \"</think>\"."
                 "Keep the answer short (e.g., \"yes\" or \"no\" for binary questions, a person's full name if "
                 "the question expects a person, a country name if it asks about a country, etc.). Output the "
                 "answer strictly after the prefix \"Answer: \"  with no extra text.")


def main():
    # TODO: Add argument parser for: model, dataset, device, hyperparams, output path, seed
    # TODO: Add experiment tracking (MLflow, Weights & Biases) to log parameters and metrics

    # TODO: Redundant - set_seed() already calls torch/np/random.seed internally
    set_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    setup_logging(logging.DEBUG)

    # TODO: No error handling for model loading (CUDA OOM, network issues, missing model files)
    # TODO: Device hardcoded - should auto-detect CUDA availability or make configurable
    # TODO: Add try-except with informative error messages
    # TODO: Consider adding model.eval() to ensure evaluation mode
    model = load_model(model_id=HF_MODEL_ID, device="cuda")
    tokenizer = load_tokenizer(model_id=HF_MODEL_ID)

    # TODO: CRITICAL - Directory management needs improvement (see shutil.rmtree concerns above)
    # TODO: Consider timestamp-based subdirectories for each run
    base_repo = DictRepo(dirpath=DICT_REPO_DIR)

    oracle = RegexOracle(fuzzy_match_threshold=0.7)

    generation_config = GenerationConfig(temperature=0.6,
                                         top_p=0.95,
                                         top_k=20,
                                         min_p=0.0,
                                         max_new_tokens=4096,
                                         pad_token_id=tokenizer.pad_token_id,
                                         do_sample=True)
    extractor = SuccessProbabilityShiftExtractor(model=model,
                                                    tokenizer=tokenizer,
                                                    oracle=oracle,
                                                    base_repo=base_repo,
                                                    prob_threshold=0.1,
                                                    num_trials=50,
                                                    min_prob=0.2,
                                                    max_prob=0.8,
                                                    batch_size=16,
                                                    generation_config=generation_config)
    df = pd.read_csv("data/hotpotqa_fullwiki_qwen3_1.7B_results.csv")

    accumulator = []
    for _, row in df.iterrows():
        pivotal_tokens = extractor.extract(reasoning_trace=row["trace"],
                                        system_prompt=SYSTEM_PROMPT,
                                        user_prompt=row["query"],
                                        expected_answer=row["ground_truth"],
                                        actual_answer="",
                                        sample_id=row["id"],
                                        metadata={})
        for pt in pivotal_tokens:
            accumulator.append({"sample_id": row["id"], **asdict(pt)})

        extractor.clear_cache()

    result_df = pd.DataFrame(accumulator)
    result_df.to_csv("data/hotpotqa_fullwiki_qwen3_1.7B_pivotal_tokens_custom.csv", index=False)


if __name__ == "__main__":
    main()
