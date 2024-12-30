"""py module for implementing the bpe tokenizer algorithm (bpe: byte-pair encoding)

original source from:
    - https://arxiv.org/pdf/1508.07909
    - https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt
"""
import re
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import AutoTokenizer


def get_stats(vocab: Dict) -> Dict:
    """ func from bpe official paper, get frequency statistic value from current vocab dictionary
    Args:
        vocab (Dict):
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs


def merge_vocab(pair, v_in: Dict):
    """ func from bpe official paper, merge the most came out pairs of subwords in input param vocab dictionary
    Args:
        pair ():
        v_in (Dict):
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def pretokenize(corpus: List[str]) -> Dict:
    """ func for applying the pre-tokenizing to input data, corpus
    Args:
        corpus (List[str]):
    """
    # for using the gpt2's pre-tokenizer algorithm, ASAP convert this logic into own implementations
    word_freq = defaultdict(int)
    pre_tokenizer = AutoTokenizer.from_pretrained("gpt2").backend_tokenizer.pre_tokenizer.pre_tokenize_str
    for text in corpus:
        words_with_offsets = pre_tokenizer(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freq[word] += 1

    return word_freq


def initialize_vocab(pretokenized: Dict) -> List:
    """ initialize the vocab structure, adding the base character in input corpus, special tokens
    current state of this func's mimic the gpt2's bpe tokenizer implementations, so we just add only one special token

    Args:
        pretokenized (Dict): dictionary for caching the freq statistic value of input data, corpus

    Return:
        initialized vocab
    """
    vocab = set()
    special_tokens = ["<|endoftext|>"]
    for word in pretokenized.keys():
        for letter in word:
            vocab.add(letter)

    return sorted(list(vocab)) + special_tokens


def get_freq_stats(freq_dict: Dict, splits: Dict) -> Dict:
    pair_freq = defaultdict(int)
    for word, freq in freq_dict.items():
        split = splits[word]
        # pass if current word is single character
        if len(split) == 1:
            continue

        for i in range(len(split)-1):
            pair = (split[i], split[i+1])
            pair_freq[pair] += freq

    return pair_freq


def merge_pair(s: str, e: str, word_freq: Dict, splits: Dict) -> Dict:
    """
    """
    for word in word_freq:
        split = splits[word]
        # pass if current word is single character
        if len(split) == 1:
            continue
        i = 0
        while i < len(split)-1:
            if split[i] == s and split[i+1] == e: split = split[:i] + [s+e] + split[i+2:]  # merge algorithm
            else: i += 1
        splits[word] = split  # change current state of word to merged one

    return splits


def tokenize(new_seq: str, merges: Dict) -> List[str]:
    """ tokenize the new context sequence by using the pretrained tokenizer, vocab
    Args:
        new_seq (str]): new single context, text sequences
    """
    # need to initialize the global space
    pre_tokenizer = AutoTokenizer.from_pretrained("gpt2").backend_tokenizer.pre_tokenizer.pre_tokenize_str


    # logic for pre-tokenizing to new context
    pre_sequence = [word for word, _ in pre_tokenizer(new_seq)]

    # logic for tokenizing to result of pre-tokenizing to new context
    splits = [[letter for letter in word] for word in pre_sequence]
    for pair, merge in merges.items():  # merge dictionary is the result of pretrained tokenizer, vocab
        s, e = pair
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split)-1:
                if split[i] == s and split[i + 1] == e: split = split[:i] + [merge] + split[i+2:]  # merge algorithm
                else: i += 1

            splits[idx] = split

    return sum(splits, [])


if __name__ == '__main__':
    # logic for building the pre tokenizer vocab
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
        pair_freq = get_freq_stats(word_freq, splits)
        best = max(pair_freq, key=pair_freq.get)
        splits = merge_pair(*best, word_freq, splits)
        new_token = "".join(best)
        merges[best] = new_token
        vocab.append(new_token)

    # logic for tokenizing the new context, text  sequence
    inference_seq = [
        "This is not a token."
    ]
    result = [tokenize(seq, merges) for seq in inference_seq]
    print(result)

