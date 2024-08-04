import re
import emoji
import nemo_text_processing

from transformers import AutoTokenizer
from nemo_text_processing.text_normalization.normalize import Normalizer


def remove_emoji(text: str) -> str:
    return re.sub(r"[\U0001F600-\U0001F64F]", "", text)  # remove emoji


def emoji2text(text: str) -> str:
    return emoji.demojize(text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s{2,}', ' ', text)


def normalize_symbol(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)


def init_normalizer(mode: str = "cased", language: str = "en") -> Normalizer:
    """ function for initializing the Text Normalizer from NVIDIA NeMo
    Args:
        mode (str): options for "lower_cased", "cased"
        language (str): default setting is english "en"

    Reference:
        https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/text_normalization/wfst/wfst_text_normalization.html#text-normalization
    """
    return Normalizer(
        input_case=mode,
        lang=language
    )


def apply_normalizer(normalizer: Normalizer, text: str) -> str:
    """ wrapper function for Text Normalizer from NVIDIA NeMo

    normalizer will do normalize tokens from written to spoken form
    e.g. 12 kg -> twelve kilograms

    normalize function's param explain:
        text: string that may include semiotic classes
        punct_pre_process: whether to perform punctuation pre-processing, for example, [25] -> [ 25 ]
        punct_post_process: whether to normalize punctuation
        verbose: whether to print intermediate meta information

    Reference:
        https://github.com/NVIDIA/NeMo-text-processing/blob/main/nemo_text_processing/text_normalization/normalize.py
        https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/text_normalization/wfst/wfst_text_normalization.html#text-normalization
    """
    return normalizer.normalize(
        text,
        verbose=False,
        punct_post_process=False
    )


def apply_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    """ wrapper function for AutoTokenizer.apply_chat_template() """
    message = [
        {
            "role": "user",
            "content": f"{prompt}"
        }
    ]
    return tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
