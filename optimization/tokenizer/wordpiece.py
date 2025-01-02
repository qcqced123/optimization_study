"""py module for implementing the word piece tokenizer algorithm
word piece algorithm is same as bpe, adding the scaler value (freq score) like as TF-IDF

original source from:
    - https://arxiv.org/pdf/1508.07909
    - https://arxiv.org/pdf/1909.03341
    - https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf
    - https://huggingface.co/learn/nlp-course/chapter6/6
    - https://www.kaggle.com/code/binfeng2021/what-is-bbpe-tokenizer-behind-llms
"""
from bpe import pretokenize
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import AutoTokenizer


def initialize_vocab(pretokenized: Dict) -> List:
    """ initialize the vocab structure, adding the base character in input corpus, special tokens
    current state of this func's mimic the gpt2's bpe tokenizer implementations, so we just add only one special token

    Args:
        pretokenized (Dict): dictionary for caching the freq statistic value of input data, corpus

    Return:
        initialized vocab
    """
    vocab = set()
    special_tokens = ["[CLS]", "[SEP]", "[UNK]", "[MASK]", "[PAD]"]
    for word in pretokenized.keys():
        prefix = word[0]
        if prefix not in vocab:
            vocab.add(prefix)

        for letter in word[1:]:
            postfix = "##" + letter
            if postfix not in vocab:
                vocab.add(postfix)

    return sorted(list(vocab)) + special_tokens


def get_scaled_freq_stats(freq_dict: Dict, splits: Dict) -> Dict:
    """ func for calculating the score, defined in word piece algorithm's original paper
    Args:
        freq_dict (Dict):
        splits (Dict):

    Math:
        score = freq(s, e)/freq(s)*freq(e)

    References:
        - https://huggingface.co/learn/nlp-course/chapter6/6
        - https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf
    """
    pair_freq = defaultdict(int)
    letter_freq = defaultdict(int)  # 이건 어디다 쓸까요
    for word, freq in freq_dict.items():
        split = splits[word]
        prefix = split[0]

        # branch for handling the single word, scoring
        if len(split) == 1:
            letter_freq[prefix] += freq
            continue

        # calculate the score for merging the most frequent sub word pair
        for i in range(len(split)-1):
            pair = (split[i], split[i+1])
            letter_freq[split[i]] += freq  # for scoring
            pair_freq[pair] += freq

        letter_freq[split[-1]] += freq

    # get score dictionary
    scores = {
        pair: freq / (letter_freq[pair[0]] * letter_freq[pair[1]]) for pair, freq in pair_freq.items()
    }
    return scores


def merge_pair(s: str, e: str, freq_dict: Dict, splits: Dict) -> Dict:
    for word in freq_dict:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == s and split[i+1] == e:
                merge = s + e[2:] if e.startswith("##") else s + e
                split = split[:i] + [merge] + split[i+2:]
            else:
                i += 1
        splits[word] = split
    return splits


def encode(text: str) -> List:
    tokens = []
    while len(text) > 0:
        i = len(text)
        while i > 0 and text[:i] not in vocab:
            i -= 1

        # unseen token handling
        if i == 0: return ["[UNK]"]

        # handling the remain sub word
        tokens.append(text[:i])
        text = text[i:]
        if len(text) > 0:
            text = "##" + text

    return tokens


def tokenize(model_name: str, text: str) -> List:
    # need to initialize the global space
    pre_tokenizer = AutoTokenizer.from_pretrained(model_name).backend_tokenizer.pre_tokenizer.pre_tokenize_str

    # logic for pre-tokenizing to new context
    pre_sequence = [word for word, _ in pre_tokenizer(text)]

    # get encoded word
    encoded = [encode(word) for word in pre_sequence]
    return sum(encoded, [])


if __name__ == '__main__':
    # logic for building the pre tokenizer vocab
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    merges = {}
    vocab_size = 70
    tokenizer_type = "wordpiece"
    model_name = "gpt2" if tokenizer_type == "bpe" else "bert-base-cased"
    word_freq = pretokenize(corpus, model_name)
    vocab = initialize_vocab(word_freq)
    splits = {word: [c if not i else "##"+c for i, c in enumerate(word)] for word in word_freq.keys()}
    while len(vocab) < vocab_size:
        scores = get_scaled_freq_stats(word_freq, splits)
        best = max(scores, key=scores.get)
        splits = merge_pair(
            *best,
            word_freq,
            splits
        )
        new_token = best[0] + best[1][2:] if best[1].startswith("##") else best[0] + best[1]
        vocab.append(new_token)

    # logic for tokenizing the new context, text sequence
    inference_seq = [
        "This is not a token.",
        "This is the Hugging Face course!"
    ]
    result = [tokenize(model_name, seq) for seq in inference_seq]
    for i in range(len(result)):
        print(result[i], end="\n")
