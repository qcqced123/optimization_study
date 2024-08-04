import re
import emoji
import nemo_text_processing
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
    """ function for initializing the Text Normalizer from NVIDIA NeMo """
    return Normalizer(
        input_case=mode,
        lang=language
    )


def apply_normalizer(normalizer: Normalizer, text: str) -> str:
    """ wrapper function for Text Normalizer from NVIDIA NeMo

    normalize function's param explain:
        text: string that may include semiotic classes
        punct_pre_process: whether to perform punctuation pre-processing, for example, [25] -> [ 25 ]
        punct_post_process: whether to normalize punctuation
        verbose: whether to print intermediate meta information
    """
    return normalizer.normalize(
        text,
        verbose=False,
        punct_post_process=False
    )
