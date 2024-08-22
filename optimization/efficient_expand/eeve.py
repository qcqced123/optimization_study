import torch
import torch.nn as nn


# number_of_old_tokens is the size of tokenizer before vocab extension.
# For example, in case of EEVE-Korean-10.8B-v1.0, number_of_old_tokens is 32000.
def freeze_partial_embedding_hook(grad):
    grad[:number_of_old_tokens] = 0
    return grad

for name, param in model.named_parameters():
    if ("lm_head" in name or "embed_tokens" in name) and "original" not in name:
        param.requires_grad = True
        if "embed_tokens" in name:
            param.register_hook(freeze_partial_embedding_hook)
    else:
        param.requires_grad = False
