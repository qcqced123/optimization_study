import torch
import torch.nn as nn

from optimum.exporters.onnx import main_export
from transformers import pipeline, BitsAndBytesConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def get_config(name: str) -> AutoConfig:
    return AutoConfig.from_pretrained(
        pretrained_model_name_or_path=name,
        trust_remote_code=True
    )


def get_tokenizer(name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=name,
        trust_remote_code=True
    )


def get_model(name: str, config: AutoConfig, bit_config: BitsAndBytesConfig = None) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=name,
        config=config,
        quantization_config=bit_config,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        torch_dtype="auto"
    )


def convert2onnx(name: str, config: AutoConfig) -> None:
    main_export(
        model_name_or_path=name,
        config=config,
        output="./saved/",
        task="text-generation-with-past",
        #opset=21,
        device="cuda",
        dtype="fp16",
        optimize="O3",
        framework="pt",
        trust_remote_code=True,
    )


if __name__ == '__main__':
    model_name = "microsoft/Phi-3-mini-128k-instruct"
    config = get_config(model_name)
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name, config)
    convert2onnx(model_name, config)
