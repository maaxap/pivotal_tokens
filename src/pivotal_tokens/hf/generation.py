import logging
import re
import typing as t
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig

from pivotal_tokens.hf.dataset import Sample


THINKING_TRACE_REGEX = r"<think>(.*?)</think>|<think>(.*)"


@dataclass(frozen=True)
class NextTokenStep:
    vocab_tuples: list[tuple[str, int, float]]
    past_key_values: t.Any | None = None


def prep_generation_context(sample: Sample, system_prompt: str, tokenizer: PreTrainedTokenizer, enable_thinking: bool) -> str:
    user_prompt = sample.query
    prompts = [{"role": "system", "content": system_prompt},
               {"role": "user", "content": user_prompt}]
    context = tokenizer.apply_chat_template(prompts,
                                            tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=enable_thinking)
    return context


def extract_thinking_trace(generated_text: str, regex: str = THINKING_TRACE_REGEX) -> str | None:
    pattern = re.compile(regex, re.DOTALL)
    match = pattern.search(generated_text)
    
    trace = None
    if match is not None:
        trace = match.group(0).strip()
    
    return trace


@torch.no_grad()
def generate_batch(samples: list[Sample],
                   system_prompt: str,
                   model: PreTrainedModel,
                   tokenizer: PreTrainedTokenizer,
                   generation_config: GenerationConfig,
                   enable_thinking: bool = True) -> list[str]:
    device = model.device
    inputs = [prep_generation_context(sample=s,
                                      system_prompt=system_prompt,
                                      tokenizer=tokenizer,
                                      enable_thinking=enable_thinking)
              for s in samples]

    enc = tokenizer(inputs, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model.generate(input_ids=input_ids,
                         attention_mask=attention_mask,
                         generation_config=generation_config)

    dec = tokenizer.batch_decode(out, skip_special_tokens=False)
    return dec


def generate_next_token(context: str,
                        model: PreTrainedModel,
                        tokenizer: PreTrainedTokenizer,
                        past_key_values: t.Any | None = None) -> NextTokenStep:
    model.eval()
    device = model.device

    enc = tokenizer(context, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    if past_key_values is None:
        # Full-context forward pass
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True)
    else:
        # Incremental step: only feed last token
        last_token = input_ids[:, -1:]
        last_mask = attention_mask[:, -1:] if attention_mask is not None else None
        out = model(input_ids=last_token,
                    attention_mask=last_mask,
                    use_cache=True,
                    past_key_values=past_key_values)

    logits = out.logits[:, -1, :]              # (1, vocab)
    logprobs = F.log_softmax(logits, dim=-1)   # (1, vocab)
    logprobs = logprobs.squeeze(0).cpu()       # (vocab,)

    vocab_size = logprobs.shape[0]
    token_ids = list(range(vocab_size))
    token_strs = tokenizer.convert_ids_to_tokens(token_ids)
    token_logprobs = logprobs.tolist()

    vocab_tuples = list(zip(token_strs, token_ids, token_logprobs))

    return NextTokenStep(vocab_tuples=vocab_tuples,
                         past_key_values=out.past_key_values)
