import random
import re
import string
from typing import Tuple

import numpy as np
import torch
from datasets import Dataset


def set_seed_for_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_label_words_and_idx(label):
    return [
        (m.group(), m.start(), m.end()) for m in re.finditer(r"(\w[\w']*\w|\w)", label)
    ]


def get_vp_ratio(text, label):
    label_words_and_idx = get_label_words_and_idx(label)
    label_words = [word for word, _, _ in label_words_and_idx]
    text_words = [text[start:end] for _, start, end in label_words_and_idx]
    vp_words = [[t, l] for t, l in zip(text_words, label_words) if t != l]
    vp_characters = [
        v
        for vp_word, label_word in vp_words
        for v, l in zip(vp_word, label_word)
        if v != l
    ]
    target_characters = [c for label_word in label_words for c in label_word]
    vp_ratio = len(vp_characters) / len(target_characters)
    return vp_ratio


def preprocess_dataset(dataset, tokenizer, max_length, num_proc):
    dataset = dataset.filter(lambda x: x["text"].strip() != "", num_proc=num_proc).map(
        lambda x: tokenizer(
            ascii_lower(x["text"])[:max_length],
            return_special_tokens_mask=True,
        ),
        num_proc=num_proc,
    )
    return dataset


def preprocess_bitcoinabuse_dataset(dataset, tokenizer, max_length, num_proc):
    # preprocess label text
    dataset = dataset.map(
        lambda x: tokenizer(
            ascii_lower(x["label"][:max_length]),
            truncation=True,
            padding=True,
        ),
        remove_columns="label",
        num_proc=num_proc,
    )
    dataset = dataset.rename_column("input_ids", "labels")

    # preprocess input text
    dataset = dataset.map(
        lambda x: tokenizer(
            ascii_lower(x["text"][:max_length]),
            truncation=True,
            padding=True,
            return_special_tokens_mask=True,
        ),
        remove_columns="text",
        num_proc=num_proc,
    )
    return dataset


def train_valid_test_split(
    ds: Dataset,
    split_ratio: Tuple[int, int, int],
    seed: int,
) -> Tuple[Dataset, Dataset, Dataset]:
    ds_train_test = ds.train_test_split(
        test_size=split_ratio[2] / sum(split_ratio), seed=seed
    )
    ds_train_valid = ds_train_test["train"].train_test_split(
        test_size=split_ratio[1] / (split_ratio[1] + split_ratio[0]), seed=seed
    )
    ds_train = ds_train_valid["train"]
    ds_valid = ds_train_valid["test"]
    ds_test = ds_train_test["test"]
    return ds_train, ds_valid, ds_test


def dict_replace(s: str, d: dict) -> str:
    d = dict((re.escape(k), v) for k, v in d.items())
    pattern = re.compile("|".join(d.keys()))
    return pattern.sub(lambda m: d[re.escape(m.group(0))], s)


def ascii_lower(sent: str) -> str:
    for u, l in zip(string.ascii_uppercase, string.ascii_lowercase):
        sent = sent.replace(u, l)
    return sent


def is_non_ascii(c: str):
    return ord(c) < 6 or ord("~") < ord(c)


def has_non_ascii(w: str):
    return any([is_non_ascii(c) for c in w])


def punycode2unicode(domain: str):
    return domain.encode().decode("idna")


def unicode2punycode(domain: str):
    return domain.encode("idna").decode()
