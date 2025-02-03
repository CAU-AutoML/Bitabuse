import re
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from datasets import load_dataset
from spellchecker import SpellChecker
from tqdm import tqdm

DATASET_NAME = "AutoML/bitabuse"
bitabuse = load_dataset(DATASET_NAME, split="train").to_pandas()
bitabuse.set_index("id", inplace=True)

spell = SpellChecker()


def spellchecker_infer(iterrow):
    idx, row = iterrow
    input_text = row["ocr"]
    label = row["label"]

    label_words_and_idx = [
        (m.group(), m.start(), m.end()) for m in re.finditer(r"(\w[\w']*\w|\w)", label)
    ]

    s = time.time()
    spell_text = input_text
    for i, (word, start, end) in enumerate(label_words_and_idx):
        input_word = input_text[start:end]
        corrected_word = spell.correction(input_word) or input_word
        if len(corrected_word) > len(input_word):
            corrected_word = corrected_word[: len(input_word)]
        elif len(corrected_word) < len(input_word):
            corrected_word += "_" * (len(input_word) - len(corrected_word))
        spell_text = spell_text[:start] + corrected_word + spell_text[end:]
    spell_time = time.time() - s
    return {"id": idx, "spellchecker": spell_text, "spellchecker_time": spell_time}


with ProcessPoolExecutor(max_workers=16) as executor:
    results = list(
        tqdm(
            executor.map(spellchecker_infer, bitabuse.iterrows(), chunksize=100),
            total=len(bitabuse),
        )
    )

result_recon = {k: [] for k in results[0].keys()}
for r in results:
    for k, v in r.items():
        result_recon[k].append(v)


bitabuse_spell = pd.DataFrame(results)
bitabuse_spell.set_index("id", inplace=True)
bitabuse_spell.to_csv("./outputs/bitabuse_infer_spell.csv")
