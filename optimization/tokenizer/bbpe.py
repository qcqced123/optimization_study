"""py module for implementing the bbpe tokenizer algorithm (bbpe: byte-level byte-pair encoding)

original source from:
    - https://arxiv.org/pdf/1508.07909
    - https://arxiv.org/pdf/1909.03341
    - https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt
    - https://www.kaggle.com/code/binfeng2021/what-is-bbpe-tokenizer-behind-llms
    - https://medium.com/@hugmanskj/understanding-tokenization-methods-a-simple-and-intuitive-guide-80c31a29f754
"""
import re
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import AutoTokenizer


if __name__ == '__main__':
    pass