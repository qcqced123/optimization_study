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
from bpe import pretokenize, initialize_vocab, merge_pair, tokenize


def get_scaled_freq_stats(freq_dict: Dict, splits: Dict) -> Dict:
    """ func for calculating the score, defined in word piece algorithm's original paper
    Args:
        pair (Tuple):
        vocab (Dict):

    Math:
        score = freq(s, e)/freq(s)*freq(e)

    References:
        - https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf
    """
    pair_freq = defaultdict(int)
    for word, freq in freq_dict.items():
        split = splits[word]
        # pass if current word is single character
        if len(split) == 1:
            continue

        for i in range(len(split)-1):
            pair = (split[i], split[i+1])
            pair_freq[pair] += freq

        # logic for calculating the scaled score

    return pair_freq


if __name__ == '__main__':
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    merges = {}
    vocab_size = 50
    word_freq = pretokenize(corpus)
    vocab = initialize_vocab(word_freq)
    splits = {word: [c for c in word] for word in word_freq.keys()}
    while len(vocab) < vocab_size:
        pair_freq = get_scaled_freq_stats(word_freq, splits)
        best = max(pair_freq, key=pair_freq.get)
        splits = merge_pair(*best, word_freq, splits)
        new_token = "".join(best)
        merges[best] = new_token
        vocab.append(new_token)

    # logic for tokenizing the new context, text sequence
    inference_seq = [
        "This is not a token."
    ]
    result = [tokenize(seq, merges) for seq in inference_seq]
    print(result)
