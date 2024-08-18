import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def set_buffer(module: nn.Module, buffer_name: str, value: torch.Tensor) -> None:
    """ set buffer for module

    Args:
        module (nn.Module): pytorch module
        buffer_name (str): e.g. "weight", "_weight", "bias" ...
        value (torch.Tensor): weight tensor

    Reference:
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
    """
    if hasattr(module, buffer_name):
        delattr(module, buffer_name)

    module.register_buffer(buffer_name, value, persistent=False)
    return


def expand_pos_emb(pos_emb: nn.Embedding, target_size: int) -> nn.Embedding:
    """ function for expanding the position embedding vector, encoding position value of more longer sequence

    Args:
         pos_emb (nn.Embedding): original position embedding vector module
         target_size (int): target size value of expanding result,
                            you must set this value bigger than pos_emb one
    """
    if pos_emb.num_embeddings >= target_size:
        raise ValueError(
            f"Target size of expanding is smaller than current position embedding's size. Please pass the bigger than current position embedding size ({pos_emb.num_embeddings})"
        )

    new_emb = nn.Embedding(target_size, pos_emb.embedding_dim, device=pos_emb.weight.device)
    num_repeats = (target_size + pos_emb.num_embeddings - 1) // pos_emb.num_embeddings  # 올림 처리
    expanded_weights = pos_emb.weight.clone().detach().repeat(num_repeats, 1)[:target_size, :]
    set_param(new_emb, "weight", nn.Parameter(expanded_weights))
    return new_emb


if __name__ == '__main__':
    # usage example code
    # filtering function for getting more good quality of data
    # use query encoder's pretrained tokenizer to calculate the sequence lengths
    model_name = "intfloat/e5-base-v2"
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto"
    )

    # expand the max sequence length of model
    # this method must be need to slightly light fine-tune
    # this method came from Longformer paper
    expand_size = 4096
    pos_emb = model.embeddings.position_embeddings
    new_emb = expand_pos_emb(pos_emb, expand_size)

    # checking
    print(f"copying original tensor is right?: {torch.eq(new_emb.weight[0], new_emb.weight[512])}")
    print(f"is grad state of expanded embedding right?: {new_emb.weight.requires_grad}")

    # set the expanded embedding layer to our model
    set_param(model.embeddings, "position_embeddings", new_emb)
    set_buffer(model.embeddings, "position_ids", torch.arange(expand_size).expand((1, -1)))
    set_buffer(
        model.embeddings,
        "token_type_ids",
        torch.zeros(model.embeddings.position_ids.size(), dtype=torch.long)
    )  # if you use bert

    text = """Lanckriet, 2007
    features is large. This emphasises SCCA’s ability to learn the semantic space from a small number of relevant features.
    In Section 2] we give a brief review of CCA, and Section [3] formulates and de- fines SCCA. In Section [4] we derive our optimisation problem and show how all the pieces are assembled to give the complete algorithm. The experiments on the paired bilingual data-sets are given in Section 5] Section [6] concludes this paper.

    2 Canonical Correlation Analysis
    We briefly review canonical correlation analysis and its ML-dual (kernel) variant to provide a smooth understanding of the transition to the sparse formulation. First, basic notation representation used in the paper is defined
    b —_ boldface lower case letters represent vectors
    s — lower case letters represent scalars
    M -— _ upper case letters represent matrices.
    The correlation between x, and x, can be computed as
    where Cag = XoX/, and Cy, = X,Xj are the within-set covariance matrices and Cy, = X_Xj is the between-sets covariance matrix, X, is the matrix whose columns are the vectors x;, i = 1,..., from the first representation while X;, is the matrix with columns x; from the second representation. We are able to observe that scaling w,, w, does not effect the quotient in equa- tion (i), which is therefore equivalent to maximising w/,C,,wy subject to WwW CaaWa = W,CwWs = 1.
    The kernelising of CCA (Fyfe & Lai, 2000a} Fyfe & Lai, 2000b) offers an al- ternative by first projecting the data into a higher dimensional feature space dt 2 X = (41,..-,%n) > O:(x) = (G1(x),..., Ow(x)) (N > n,t =a,b) be fore performing CCA in the new feature spaces. The kernel variant of CCA
    is useful when the correlation is believed to exist in some non linear relation- ship. Given the kernel functions kK, and ky let Ka = X/Xq and Ky = X},Xp be the linear kernel matrices corresponding to the two representations of the data, where X, is now the matrix whose columns are the vectors a(xi), i=1,...,¢from the first representation while X; is the matrix with columns o»(x:) from the second representation. The weights w, and wy, can be ex- pressed as a linear combination of the training examples wz = X,a and w, = X,3. Substitution into the ML-primal CCA equation (f) gives the optimisation
    which is equivalent to maximising a’ K,K,B subject to a’ K2a = B'K2B = 1. This is the ML-dual form of the CCA optimisation problem given in equa- tion (I) which can be cast as a generalised eigenvalue problem and for which the first & generalised eigenvectors can be found efficiently. Both CCA and KCCA can be formulated as symmetric eigenproblems.
    A variety of theoretical analyses have been presented for CCA (Akaho, 2001} A common conclusion of some of these analyses is the need to regularise KCCA. For example the quality of the generalisation of the associated pat- tern function is shown to be controlled by the sum of the squares of the weight vector norms in (Hardoon & Shawe-Taylor, In Press). Although there are advantages in using KCCA, which have been demonstrated in various ex- periments across the literature, we clarify that when using a linear kernel in both views, regularised KCCA is the same as regularised CCA (since the former and latter are linear). Nonetheless using KCCA with a linear kernel can have advantages over CCA, the most important being speed when the number of features is larger than the number of samples|}|

    3 Sparse CCA
    The motivation for formulating a ML primal-dual SCCA is largely intuitive when faced with real-world problems combined with the need to understand or interpret the found solutions. Consider the following examples as potential

    Bach"""
    inputs = tokenizer(text, return_tensors="pt")

    feature = model(
        input_ids=inputs["input_ids"],
        token_type_ids=inputs["token_type_ids"],
        attention_mask=inputs["attention_mask"]
    )
    last_hidden_state, cls_pooling = feature["last_hidden_state"], feature["pooler_output"]
    print(f"shape of last hidden state from expanded max_sequence length model is: {last_hidden_state.shape}")
    print(f"shape of cls pooling output from expanded max_sequence length model is: {cls_pooling.shape}")
