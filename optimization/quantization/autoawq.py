""" AWQ: Activation Aware Quantization Tutorial
This tutorial have the objective for using AWQ Quantized model to LoRA fine-tune pipeline

workflow:
    1) load the pretrained model
    2) apply the Quantization with AWQ
    3) save the applied AWQ model
    4) re-load the AWQ Model with transformers.AutoModel
    5) add the AWQ AutoModel to LoRA adapter
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
from peft import LoraConfig, PeftModel
from peft import get_peft_config, get_peft_model

from awq import AutoAWQForCausalLM
from awq.models.base import BaseAWQForCausalLM
from transformers import AwqConfig, BitsAndBytesConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_tokenizer(model_name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )


def get_awq_model(model_name: str) -> BaseAWQForCausalLM:
    return AutoAWQForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )


def get_bit_config() -> Dict:
    return {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }


def do_quantize(model: nn.Module, tokenizer: AutoTokenizer, q_config: Dict) -> None:
    model.quantize(
        tokenizer,
        quant_config=q_config
    )
    return


def modify_quant_config(model: nn.Module, q_config: Dict) -> None:
    modified_config = AwqConfig(
        bits=q_config["w_bit"],
        group_size=q_config["q_group_size"],
        zero_point=q_config["zero_point"],
        version=q_config["version"].lower(),
    ).to_dict()
    model.model.config.quantization_config = modified_config
    return


def save_awq_module(model: nn.Module, tokenizer: AutoTokenizer, path: str) -> None:
    model.save_quantized(path)
    tokenizer.save_pretrained(path)
    return


if __name__ == '__main__':
    device = get_device()
    model_name = "microsoft/Phi-3-mini-128k-instruct"

    tokenizer = get_tokenizer(model_name=model_name)
    model = get_awq_model(model_name)
    bit_config = get_bit_config()

    do_quantize(
        model=model,
        tokenizer=tokenizer,
        q_config=bit_config
    )

    modify_quant_config(
        model=model,
        q_config=bit_config
    )

    save_awq_module(
        model=model,
        tokenizer=tokenizer,
        path="./awq/phi3"
    )

