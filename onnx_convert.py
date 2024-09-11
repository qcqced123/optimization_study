import torch.nn as nn

from optimum.exporters.onnx import main_export
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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


def print_model_param_info(model: nn.Module) -> None:
    """print the name, data types, require_grad info of the parameters for each module in the given PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.
    """
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            print(
                f"Module: {name}, Parameter: {param_name}, DataType: {param.dtype}, is_trainable: {param.requires_grad}")


def convert2onnx(name: str) -> None:
    """ function to export the pytorch model into ONNX Format with optimum

    Args:
        name (str): path for local hub or model name in huggingface remote model hub
    """
    main_export(
        model_name_or_path=name,
        output="./saved/",
        task="text-generation-with-past",
        opset_version=21,
        device="cpu",
        dtype="fp32",
        framework="pt",
        trust_remote_code=True,
    )


if __name__ == '__main__':
    model_name = "./saved/stage5-eeve-phi3.5-mini-instruct/"
    config = get_config(model_name)
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name, config)

    print_model_param_info(model)
    # convert2onnx(model_name, config)
