import os
import re

import torch
import torch.nn as nn

from torch import Tensor
from typing import Union, Callable, List, Tuple, Set, Dict
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, pipeline, TrainingArguments, Trainer
os.environ["LRU_CACHE_CAPACITY"] = "4096"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8, expandable_segments:True"


def is_korean(token: str) -> bool:
    """ bool function, checking input token if it is korean
    return "True" when the token string have the korean char

    Args:
        token (str): input token for checking korean
    """
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
# For example, in case of Phi3.5-mini-instruct, number_of_old_tokens is 32011
def freeze_partial_embedding_hook(
    module: nn.Module,
    value: Tensor,
    nums_of_original_embed: int
) -> None:
    """ freeze the sub-part of embedding or language modeling layer's weight, register this function to backward hook

    Args:
        module (nn.Module): torch layer module of freezing
        value (torch.Tensor): embedding or lm head weight tensor
        nums_of_original_embed (int): value of original (adding before new tokens) vocab size

    Example Result (freeze part: 0 ~ 2, fire part: 3 ~ 5):

        before module's weight value:
        tensor([[-9.5572e-01, -9.7061e-01, -8.1656e-01,  7.8150e-01],
                [-1.1405e+00, -1.2238e+00,  3.1620e-01,  7.9152e-01],
                [-4.5088e-01,  1.2617e-01, -2.0796e+00, -8.1123e-01],
                [-4.4096e-01,  3.3281e-01,  9.0054e-02,  1.2089e-01],
                [ 6.5785e-01,  1.0774e+00,  1.3995e-03,  1.0219e+00],
                [-2.7588e-01, -6.3284e-01, -1.4350e+00, -1.1939e+00]])

        after module's weight value:
        tensor([[-0.9557, -0.9706, -0.8166,  0.7815],
                [-1.1405, -1.2238,  0.3162,  0.7915],
                [-0.4509,  0.1262, -2.0796, -0.8112],
                [-0.1701, -0.1202,  1.6909, -0.7109],
                [-0.2855,  0.2742, -1.0133,  0.4578],
                [ 1.5339,  0.0960,  1.6975,  0.0354]])

    Reference:
        https://arxiv.org/pdf/2402.14714
        https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0
        https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html
        https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html
    """
    # split the weight into two apart
    # detach() will copy the tensor and remove its computational graph
    # detach() result does not affect the original tensor
    freeze_part, new_part = value[:nums_of_original_embed].detach(), value[nums_of_original_embed:]

    # for making the only "new_part" weight updated
    # freeze_part and new_part do not have same "requires_grad" state
    # so if we concatenate both weight naively in this case,
    # this behavior will be caused the error (can't optimize the non-leaf tensor)
    with torch.no_grad():
        cnt_emb = torch.cat([freeze_part, new_part])

    set_param(module, "weight", cnt_emb)
    return


def set_train_layer(model: nn.Module, stage: int) -> None:
    """ set the trainable layer for training by EEVE method

    behavior in each stage:
        --------------------------------------------------------------
        stage 1. train only added word embedding
        stage 2. train only added language modeling head
        stage 3. train only stage 1. and stage 2.
        --------------------------------------------------------------
        stage 4. train full language modeling head
        stage 5. train added word embedding and full language modeling head
        stage 6. train full layers
        stage 7. train only decoder layer
        --------------------------------------------------------------
    Args:
        model (nn.Module):
        stage (int): value of current stage, this value will determine the current trainable layer

    Reference:
        https://arxiv.org/pdf/2402.14714
        https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0
    """
    # define the stage to need partial training
    # get the target modules of each stage
    stage_dict = {
        1: ["partial-embed_tokens"],
        2: ["partial-lm_head"],
        3: ["partial-embed_tokens", "partial-lm_head"],
        4: ["full-lm_head"],
        5: ["partial-embed_tokens", "full-lm_head"],
        6: [""],
        7: [""],
    }
    target_modules = stage_dict[stage]

    # set the requires_grad options of each stage, modules
    for target in target_modules:
        state, target_name = target.split('-')

        for name, module in model.named_modules():
            for param_name, param in module.named_parameters():
                if target_name in name and state == "partial":
                    freeze_partial_embedding_hook(
                        module=module,
                        value=param,
                        nums_of_original_embed=32011,
                    )

                elif target_name not in name:  # for not current train-stage layer
                    param.requires_grad = False
    return
