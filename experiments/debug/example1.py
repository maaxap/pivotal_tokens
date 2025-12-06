import logging
import shutil
from pathlib import Path

from transformers import GenerationConfig

from pivotal_tokens.hf import load_model, load_tokenizer
from pivotal_tokens.extractor import SuccessProbabilityShiftExtractor
from pivotal_tokens.oracle import RegexOracle
from pivotal_tokens.repo import DictRepo
from pivotal_tokens.utils import setup_logging


HF_MODEL_ID = "Qwen/Qwen3-1.7B"
DICT_REPO_DIR = Path("repo")
ORACLE_ANSWER_REGEX = r"(?s)\s*(?:Answer:\s*)?(.*?)\s*(?=(?:<\|im_end\|>|<\|endoftext\|>|\Z))"

SYSTEM_PROMPT = ("Answer the following question directly, strictly without chain-of-thought after \"</think>\"."
                 "Keep the answer short (e.g., \"yes\" or \"no\" for binary questions, a person's full name if "
                 "the question expects a person, a country name if it asks about a country, etc.). Output the "
                 "answer strictly after the prefix \"Answer: \"  with no extra text.")


def main():
    setup_logging(logging.DEBUG)

    shutil.rmtree(DICT_REPO_DIR)

    model = load_model(model_id=HF_MODEL_ID, device="cpu")
    tokenizer = load_tokenizer(model_id=HF_MODEL_ID)

    base_repo = DictRepo(dirpath=DICT_REPO_DIR)
    oracle = RegexOracle(answer_regex=ORACLE_ANSWER_REGEX,
                            fuzzy_match_threshold=0.5)

    generation_config = GenerationConfig(temperature=0.6,
                                            top_p=0.95,
                                            top_k=20,
                                            min_p=0.1,
                                            max_new_tokens=32,
                                            do_sample=True)
    extractor = SuccessProbabilityShiftExtractor(model=model,
                                                    tokenizer=tokenizer,
                                                    oracle=oracle,
                                                    base_repo=base_repo,
                                                    prob_threshold=0.05,
                                                    num_trials=2,
                                                    min_prob=0.1,
                                                    max_prob=0.9,
                                                    batch_size=2,
                                                    generation_config=generation_config) 

    reasoning_trace = ("The capital of Czech Republic is Paris. Wait, I'm wrong, it is Prague.")
    expected_answer = "Prague"
    user_prompt = "What is the capital of Czech Republic?"
    metadata = {"example_id": "debug_001"}

    pivotal_tokens = extractor.extract(reasoning_trace=reasoning_trace,
                                       system_prompt=SYSTEM_PROMPT,
                                       user_prompt=user_prompt,
                                       expected_answer=expected_answer,
                                       actual_answer="Prague",
                                       sample_id="example_1",
                                       metadata=metadata)
    print("Extracted Pivotal Tokens:", pivotal_tokens)


if __name__ == "__main__":
    main()
