import os
import gc
import copy
import json
import shutil

import torch
import torch.nn as nn
import bitsandbytes as bnb

from tqdm.auto import tqdm
from typing import Dict, List

from peft.utils import _get_submodules
from peft import PeftModel, LoraConfig
from peft import get_peft_config, get_peft_model
from bitsandbytes.functional import dequantize_4bit

from transformers import BitsAndBytesConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )


def get_config(model_name: str) -> AutoConfig:
    return AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True
    )


def get_bit_config() -> BitsAndBytesConfig:
    """ function for getting QLoRA bit config """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )


def get_qlora_model(model_name: str, config: AutoConfig, bit_config: BitsAndBytesConfig, device: str) -> nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=bit_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map=device
    )


def get_lora_config() -> LoraConfig:
    return LoraConfig(
        target_modules="all-linear",
        task_type="None",
        inference_mode=True,
        r=8,  # rank value
        lora_alpha=32,
        lora_dropout=0.1,
        bias='none',
    )


def apply_qlora(model: nn.Module, lora_config: LoraConfig) -> PeftModel:
    return PeftModel.from_pretrained(
        model=model,
        peft_config=lora_config
    )


def save_model(model: nn.Module, tokenizer: AutoTokenizer, to: str) -> None:
    print(f"Saving dequantized model to {to}...")
    model.save_pretrained(to)
    tokenizer.save_pretrained(to)
    config_data = json.loads(open(os.path.join(to, 'config.json'), 'r').read())
    config_data.pop("quantization_config", None)
    config_data.pop("pretraining_tp", None)
    with open(os.path.join(to, 'config.json'), 'w') as config:
        config.write(json.dumps(config_data, indent=2))


def dequantize_model(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    to: str = './dequantized_model',
    dtype = torch.bfloat16,
    device: str = "cuda:0"
) -> nn.Module:
    """
    'model': the peftmodel you loaded with qlora.
    'tokenizer': the model's corresponding hf's tokenizer.
    'to': directory to save the dequantized model
    'dtype': dtype that the model was trained using
    'device': device to load the model to
    """

    # Delete the model object if it exists
    if os.path.exists(to):
        shutil.rmtree(to)

    os.makedirs(to, exist_ok=True)
    cls = bnb.nn.Linear4bit
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                quant_state = copy.deepcopy(module.weight.quant_state)
                quant_state.dtype = dtype
                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)

                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)

                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)

        model.is_loaded_in_4bit = False
        save_model(model, tokenizer, to)
        return model


def merge_llm_with_lora(peft_model: PeftModel) -> PeftModel:
    return peft_model.merge_and_unload()


if __name__ == '__main__':
    """ initialize the necessary module, do QLoRA Quantize """
    device = get_device()
    model_name = "microsoft/Phi-3-mini-128k-instruct"

    tokenizer = get_tokenizer(model_name)
    config = get_config(model_name)
    bit_config = get_bit_config()

    model = get_qlora_model(
        model_name=model_name,
        config=config,
        bit_config=bit_config,
        device="cuda:0",
    )

    dequantize_model = dequantize_model(
        model=model,
        tokenizer=tokenizer
    )

    """ do de-quantize QLoRA to fp16, merge LLM and LoRA weight """
    lora_config = get_lora_config()

    qlora_model = get_peft_model(
        model=dequantize_model,
        peft_config=lora_config,
    )

    merged_model = merge_llm_with_lora(peft_model=qlora_model)