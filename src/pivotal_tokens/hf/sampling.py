import logging
import re

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig

from pivotal_tokens.hf.dataset import Sample


THINKING_TRACE_REGEX = r"<think>(.*?)</think>|<think>(.*)"


def prep_generation_context(sample: Sample, system_prompt: str, tokenizer: PreTrainedTokenizer) -> str:
    user_prompt = sample.query
    prompts = [{"role": "system", "content": system_prompt},
               {"role": "user", "content": user_prompt}]
    context = tokenizer.apply_chat_template(prompts,
                                            tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=True)
    return context


def extract_thinking_trace(generated_text: str, regex: str = THINKING_TRACE_REGEX) -> str:
    pattern = re.compile(regex, re.DOTALL)
    match = pattern.search(generated_text)
    
    trace = None
    if match is not None:
        trace = match.group(0).strip()
    
    return trace


def batch_sample(
    samples: list[Sample],
    system_prompt: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    generation_config: GenerationConfig,
    batch_size: int | None = None
) -> list[str]:
    
    if batch_size is None:
        batch_size = len(samples)

    inputs = []
    for sample in samples:
        context = prep_generation_context(sample=sample,
                                          system_prompt=system_prompt,
                                          tokenizer=tokenizer)
        inputs.append(context)
    
    encodings = tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)

    outputs = []
    remaining_samples = len(samples)
    with torch.no_grad():
        while remaining_samples > 0:
            current_batch_size = min(batch_size, remaining_samples)
            logging.debug(f"Remaining samples to generate: {remaining_samples}/{len(samples)}")

            batch_outputs = model.generate(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           generation_config=generation_config)
            outputs.extend(batch_outputs)

            remaining_samples -= current_batch_size

    
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return decoded_outputs 
    