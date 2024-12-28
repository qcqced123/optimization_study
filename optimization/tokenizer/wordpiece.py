"""py module for implementing the word piece tokenizer algorithm

original source from:
    - https://arxiv.org/pdf/1508.07909
    - https://arxiv.org/pdf/1909.03341
    - https://huggingface.co/learn/nlp-course/chapter6/6
    - https://www.kaggle.com/code/binfeng2021/what-is-bbpe-tokenizer-behind-llms
"""
import re
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
