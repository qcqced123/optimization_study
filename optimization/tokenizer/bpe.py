"""py module for implementing the bpe tokenizer algorithm (bpe: byte-pair encoding)

original source from:
    - https://arxiv.org/pdf/1508.07909
    - https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt
"""
import re
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import AutoTokenizer


# def get_stats(vocab):
#     pairs = defaultdict(int)
#     for word, freq in vocab.items():
#         symbols = word.split()
#         for i in range(len(symbols)-1):
#             pairs[symbols[i],symbols[i+1]] += freq
#     return pairs
#
#
# def merge_vocab(pair, v_in):
#     v_out = {}
#     bigram = re.escape(' '.join(pair))
#     p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
#     for word in v_in:
#         w_out = p.sub(''.join(pair), word)
#         v_out[w_out] = v_in[word]
#     return v_out


def init_vocab(seq: str) -> Dict:
    """
    Args:
        seq (str): sequence from corpus, data
    """
    vocab = defaultdict(int)
    for s in seq.split():
        for char in list(s):
            vocab[char] += 1

    return vocab


def get_char_stats(vocab: Dict) -> Dict:
    """
    Args:
        vocab (Dict): vocab dictionary by splitting from corpus, data
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        chars = word.split()
        for i in range(len(chars)-1):
            pairs[chars[i],chars[i+1]] += freq

    return pairs


def add_vocab(best_pair: Tuple, vocab: Dict):
    """
    Args:
        best_pair (Tuple):
        vocab (Dict):
    """
    new_vocab = defaultdict(int)
    bi_gram = re.escape("".join(best_pair))
    p = re.compile(r'(?<!\S)' + bi_gram + r'(?!\S)')
    for word in vocab:
        word_out = p.sub("".join(best_pair), word)
        new_vocab[word_out] = vocab[word]

    return new_vocab


if __name__ == '__main__':
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    word_freqs = defaultdict(int)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

    alphabet = []

    for word in word_freqs.keys():
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    alphabet.sort()

    vocab = ["<|endoftext|>"] + alphabet.copy()
    splits = {word: [c for c in word] for word in word_freqs.keys()}
    print(splits)
    # vocab = init_vocab(text)
    # vocab_size = 10
    # for i in range(vocab_size):
    #     pairs = get_char_stats(vocab)
    #     print(pairs)
    #     best = max(pairs, key=pairs.get)
    #     vocab = add_vocab(best, vocab)
    #
    # for k,v in vocab.items():
    #     print(f"{k}: {v}", end="\n")

    # vocab = {'l o w </w>': 5, 'l o w e r </w>': 2,
    #          'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
    # num_merges = 10
    # for i in range(num_merges):
    #     pairs = get_stats(vocab)
    #     best = max(pairs, key=pairs.get)
    #     vocab = merge_vocab(best, vocab)
    #     print(best)
    #     print(vocab)