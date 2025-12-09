import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pivotal_tokens.constants import get_hf_cache_dir



def load_model(model_id: str, device: str):

    use_flash_attention = False
    try:
        import flash_attn

        logging.info("Flash Attention 2 is available and will be used")
        use_flash_attention = True
    except ImportError:
        logging.info("Flash Attention 2 is not available, using standard attention")


    # Add flash attention to config if available
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device,
        "cache_dir": str(get_hf_cache_dir())
    }

    if use_flash_attention:
        logging.debug("Configuring model to use Flash Attention 2")
        # Flash Attention requires either float16 or bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            # Use bfloat16 for Ampere or newer GPUs (compute capability 8.0+)
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            # Use float16 for older GPUs
            model_kwargs["torch_dtype"] = torch.float16

        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    return model


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=get_hf_cache_dir(), trust_remote_code=True)

    tokenizer.padding_side = "left"

    # Set padding token if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer
