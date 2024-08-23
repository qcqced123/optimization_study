import os
import re
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Union, Callable, List, Tuple, Set, Dict
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, pipeline, TrainingArguments, Trainer
os.environ["LRU_CACHE_CAPACITY"] = "4096"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8, expandable_segments:True"


def is_korean(token: str) -> bool:
    """ bool function, checking input token if it is korean """
    # return "True" when the token string have the korean char
    if re.search(r'[\uAC00-\uD7A3]', token):
        return True

    return False


def get_subwords_indices(tokenizer: AutoTokenizer, new_tokens: Union, new_tokens_indices: List[int]) -> Dict:
    """ function for returning the subwords indices of new_tokens from original tokenizer (before adding the new_tokens)
    Args:
        tokenizer (AutoTokenizer): Tokenizer before adding the new_tokens
        new_tokens (Union, List): Iterable object of containing the new_tokens
        new_tokens_indices (List): List object of containing the new_tokens indices in new tokenizer (after adding the new_tokens)
    """
    return {k: tokenizer(v).input_ids for k,v in zip(new_tokens_indices, new_tokens)}


def add_new_tokens(tokenizer: AutoTokenizer, new_tokens: Union) -> None:
    """ add the new tokens of target language
    In our case, we target the korean
    """
    tokenizer.add_tokens(
        new_tokens=new_tokens,
        special_tokens=False
    )


def get_subwords_emd(word_emd: Tensor, indices_dict: Dict[int, List]) -> Tensor:
    """ get the subwords word embedding value
    Args:
        word_emd (Tensor): weight tensor of model's word embedding (before adding the new_tokens)
        indices_dict (Dict): new_tokens's subwords's indices dictionary from get_subwords_indcies()
    """
    return torch.vstack([word_emd[v, :].mean(dim=0) for k, v in indices_dict.items()]).detach()


def set_param(module: nn.Module, param_name: str, value: torch.Tensor) -> None:
    """ set param for module or set the new module for whole model architecture

    Args:
        module (nn.Module): pytorch module
        param_name (str): e.g. "weight", "_weight", "bias" ...
        value (torch.Tensor): weight tensor

    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
    """
    if hasattr(module, param_name):
        delattr(module, param_name)

    setattr(module, param_name, value)
    return


# number_of_old_tokens is the size of tokenizer before vocab extension.
# For example, in case of Phi3.5-mini-instruct, number_of_old_tokens is 32010
def freeze_partial_embedding_hook(module: Tensor, nums_of_original_embed: int) -> Tensor:
    """ freeze the sub-part of embedding layer & lm head, register this function to backward hook
    Args:
        module (nn.Module): torch module's weight of subpart freezing
        nums_of_original_embed (int): value of original (adding before new tokens) vocab size

    Reference:
        https://arxiv.org/pdf/2402.14714
        https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0
        https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html
    """
    module[:nums_of_original_embed].require_grad = False
    return module


def set_train_layer() -> None:
    """ set the trainable layer for training by EEVE method
    """
    return


for name, param in model.named_parameters():
    if ("lm_head" in name or "embed_tokens" in name) and "original" not in name:
        param.requires_grad = True
        if "embed_tokens" in name:
            param.register_hook(freeze_partial_embedding_hook)
    else:
        param.requires_grad = False
