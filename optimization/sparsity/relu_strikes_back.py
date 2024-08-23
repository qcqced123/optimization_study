import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Dict, Set


def get_proj_module_names(model: nn.Module) -> Set:
    """ function for getting the linear module's (projection layer) name,
    using the passing it to the lora config's param named "target_module" when you use the AWQ Quantization Strategy

    Peft.LoraConfig does not detect the AWQ's model module
    for example, AWQ's model named parameter "qkv_proj" have the linear module, named "WQLinear", quite different format from pytorch as nn.Linear
    in this case, Peft does not detect "qkv_proj", so it might cause the error.

    so this function manually find the Linear Module(WQLinear) from AWQ Model and return the it's name set

    Args:
        model (nn.Module): target model's instance

    Returns:
        {q_proj, v_proj, ... down_proj}
    """
    proj_module_names = set()
    for name, module in model.named_modules():
        if "proj" in name:
            for sub_name in name.split('.'):
                if sub_name.endswith("proj"):
                    proj_module_names.add(sub_name)

    return proj_module_names


def print_model_param_info(model: nn. Module):
    """ print function the name, data types, require_grad info of the parameters
    for each module in the given pytorch model

    Args:
        model (nn.Module): The PyTorch model.
    """
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            print(
                f"Module: {name}, Parameter: {param_name}, DataType: {param.dtype}, is_trainable: {param.requires_grad}")


def convert_activation_func(model: nn.Module) -> None:
    """ convert non-ReLU activation function to ReLU in feed-forward network

    Args:
        model (nn.Module): LLM module to convert activation function
    """
    for name, module in model.named_modules():
        if "activation" in name:
            print(f"name: {name}, module: {module}")



