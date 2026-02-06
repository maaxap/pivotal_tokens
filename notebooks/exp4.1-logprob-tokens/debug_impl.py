# %%
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from pivotal_tokens.extractor import (
    THINKING_END_TOKEN,
    LogLikelihoodSpikeExtractor,
)
from pivotal_tokens.hf.loading import load_model, load_tokenizer
from pivotal_tokens.repo import DictRepo
from pivotal_tokens.utils import setup_logging

# %%
QWEN3_1_7B_MODEL_ID = "Qwen/Qwen3-1.7B"
DEVICE = "cuda"

# %%
setup_logging(level=logging.DEBUG)

# %%
# Dummy inputs
SAMPLE_ID = "debug_sample_001"
SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT = "What is 2 + 2?"
REASONING_TRACE = "I add the numbers. 2 + 2 = 4."
ACTUAL_ANSWER = "4"

print("Sample ID:", SAMPLE_ID)
print("System prompt:", SYSTEM_PROMPT)
print("User prompt:", USER_PROMPT)
print("Reasoning trace:", REASONING_TRACE)
print("Actual answer:", ACTUAL_ANSWER)

# %%
# Load model/tokenizer
model = load_model(model_id=QWEN3_1_7B_MODEL_ID, device=DEVICE)
tokenizer = load_tokenizer(model_id=QWEN3_1_7B_MODEL_ID)
model.eval()

print("Model loaded:", QWEN3_1_7B_MODEL_ID)
print("Tokenizer vocab size:", len(tokenizer))
print("Device:", DEVICE)

# %%