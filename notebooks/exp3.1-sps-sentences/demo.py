# %% [markdown]
# Demo for sentence-level SuccessProbabilityShiftSentenceExtractor.
# It prints the outputs of the key methods so you can see the intermediate results.

# %%
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from transformers import GenerationConfig

from pivotal_tokens.extractor import (
    THINKING_END_TOKEN,
    SuccessProbabilityShiftSentenceExtractor,
)
from pivotal_tokens.hf.loading import load_model, load_tokenizer
from pivotal_tokens.oracle import RegexOracle
from pivotal_tokens.repo import DictRepo, SampleRepo
from pivotal_tokens.utils import setup_logging

# %%
CONFIG_PATH = Path(
    "/home/xaparo00/workspace/projects/pivotal_tokens/data/experiments/exp2.1-sps-tokens/exp2.1.4-qwen3-1.7b-sps-tokens-small-prob-threshold/config.json"
)
SAMPLE_INDEX = 0
MAX_SENTENCES_FOR_DEMO = 4
MAX_ITEMS_TO_PRINT = 6
PREVIEW_CHARS = 1000000000

# Optional overrides for quick runs. Set to None to use config values.
OVERRIDE_NUM_TRIALS = None
OVERRIDE_BATCH_SIZE = None


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def strip_thinking_end(trace: str) -> str:
    if not trace:
        return ""
    trimmed = trace.rstrip()
    if trimmed.endswith(THINKING_END_TOKEN):
        trimmed = trimmed[: -len(THINKING_END_TOKEN)].rstrip()
    return trimmed


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    safe_text = (text or "").replace("\n", "\\n")
    if len(safe_text) <= limit:
        return safe_text
    return safe_text[:limit].rstrip() + "..."


def print_sentences(title: str, sentences: list[str], max_items: int = MAX_ITEMS_TO_PRINT) -> None:
    print(f"{title}: {len(sentences)} sentence(s)")
    for idx, sentence in enumerate(sentences[:max_items], start=1):
        print(f"  {idx:02d}. {preview(sentence)}")
    if len(sentences) > max_items:
        print(f"  ... {len(sentences) - max_items} more")


def print_segments(title: str, segments: list[list[str]], max_items: int = MAX_ITEMS_TO_PRINT) -> None:
    print(f"{title}: {len(segments)} segment(s)")
    for idx, segment in enumerate(segments[:max_items], start=1):
        segment_text = "".join(segment)
        print(f"  {idx:02d}. len={len(segment)} text={preview(segment_text)}")
    if len(segments) > max_items:
        print(f"  ... {len(segments) - max_items} more")


# %%
config = load_config(CONFIG_PATH)
setup_logging(level=logging.DEBUG if config.get("debug") else logging.INFO)

trace_path = "/home/xaparo00/workspace/projects/pivotal_tokens/data/artifacts/exp1.1.1-qwen3-1.7b-traces.csv"
traces_df = pd.read_csv(trace_path)

sample_row = traces_df.iloc[SAMPLE_INDEX]
sample_id = str(sample_row["id"])
user_prompt = str(sample_row["query"])
expected_answer = str(sample_row["ground_truth"])
reasoning_trace = strip_thinking_end(str(sample_row["trace"]))
system_prompt = str(config["system_prompt"])

print("Sample ID:", sample_id)
print("User prompt:", preview(user_prompt))
print("Expected answer:", expected_answer)
print("Reasoning trace preview:", preview(reasoning_trace))
print("Traces file:", trace_path)

# %%
model = load_model(model_id=config["model_id"], device=config["device"])
tokenizer = load_tokenizer(model_id=config["model_id"])
model.eval()

# %% 
generation_config = GenerationConfig(
    max_new_tokens=config["max_new_tokens"],
    temperature=config["temperature"],
    top_p=config["top_p"],
    top_k=config["top_k"],
    min_p=config["min_p"],
    do_sample=config["do_sample"],
    pad_token_id=tokenizer.pad_token_id,
)

oracle = RegexOracle(fuzzy_match_threshold=config["fuzzy_match_threshold"])

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
repo_dir = Path("/home/xaparo00/workspace/projects/pivotal_tokens/data/demo/{run_id}")
base_repo = DictRepo(dirpath=repo_dir)
sample_repo = SampleRepo(sample_id=sample_id, base_repo=base_repo)

num_trials = OVERRIDE_NUM_TRIALS if OVERRIDE_NUM_TRIALS is not None else config["num_trials"]
batch_size = OVERRIDE_BATCH_SIZE if OVERRIDE_BATCH_SIZE is not None else config["batch_size"]

extractor = SuccessProbabilityShiftSentenceExtractor(
    model=model,
    tokenizer=tokenizer,
    oracle=oracle,
    base_repo=base_repo,
    prob_threshold=config["prob_threshold"],
    num_trials=num_trials,
    min_prob=config["min_prob"],
    max_prob=config["max_prob"],
    batch_size=batch_size,
    generation_config=generation_config,
)

print("Repo dir:", repo_dir)
print("Num trials:", num_trials)
print("Batch size:", batch_size)

# %%
context = extractor.create_context(system_prompt=system_prompt, user_prompt=user_prompt)
print("create_context preview:", )
print(context)

# %%

print(reasoning_trace)
# %%
sentences = extractor.split_into_sentences(reasoning_trace)
print("\n***\n".join(sentences))


# %%
demo_sentences = (
    sentences if MAX_SENTENCES_FOR_DEMO is None else sentences[:MAX_SENTENCES_FOR_DEMO]
)

print("\n\n".join(demo_sentences))

# %%
left, right = extractor.split_sentence_sequence(demo_sentences)
print("\n\n".join(left))

print(" * * * ")

print("\n\n".join(right))

# %%
init_prob = extractor.estimate_success_probability(
    prefix="",
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    expected_answer=expected_answer,
    sample_repo=sample_repo,
    metadata={"stage": "demo_init_probability"},
)
print("estimate_success_probability (empty prefix):", init_prob)

first_prefix = demo_sentences[0]
first_prob = extractor.estimate_success_probability(
    prefix=first_prefix,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    expected_answer=expected_answer,
    sample_repo=sample_repo,
    metadata={"stage": "demo_first_sentence"},
)
print("estimate_success_probability (first sentence prefix):", first_prob)

# %%
subdivided = extractor.subdivide_sentence_sequence(
    sequence=demo_sentences,
    prefix=[],
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    expected_answer=expected_answer,
    sample_repo=sample_repo,
)
for s in subdivided:
    print(s)

# %%
pivotal_spans = extractor.extract(
    sample_id=sample_id,
    reasoning_trace=reasoning_trace,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    actual_answer="",
    expected_answer=expected_answer,
    metadata={"demo": True},
)

print(f"extract -> {len(pivotal_spans)} span(s)")
for idx, span in enumerate(pivotal_spans[:MAX_ITEMS_TO_PRINT], start=1):
    print(
        f"  {idx:02d}. pivotal={span.is_pivotal} "
        f"delta={span.prob_delta:.4f} "
        f"before={span.prob_before:.4f} after={span.prob_after:.4f} "
        f"text={span.span_text}"
    )
    print("       metadata:", {k: v for k, v in (span.metadata or {}).items()})

if len(pivotal_spans) > MAX_ITEMS_TO_PRINT:
    print(f"  ... {len(pivotal_spans) - MAX_ITEMS_TO_PRINT} more")

# %%
if pivotal_spans:
    print("First span as dict:")
    print(asdict(pivotal_spans[0]))

# %%
