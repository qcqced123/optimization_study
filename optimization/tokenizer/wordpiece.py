"""py module for implementing the word piece tokenizer algorithm
word piece algorithm is same as bpe, adding the scaler value (freq score) like as TF-IDF

original source from:
    - https://arxiv.org/pdf/1508.07909
    - https://arxiv.org/pdf/1909.03341
    - https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf
    - https://huggingface.co/learn/nlp-course/chapter6/6
    - https://www.kaggle.com/code/binfeng2021/what-is-bbpe-tokenizer-behind-llms
"""
import re
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import AutoTokenizer


def calculate_score(pair: Tuple, vocab: Dict):
    """ func for calculating the score, defined in word piece algorithm's original paper
    Args:
        pair (Tuple):
        vocab (Dict):

    Math:
        score = freq(s, e)/freq(s)*freq(e)

    References:
        - https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf
    """
    return


if __name__ == '__main__':
    pass