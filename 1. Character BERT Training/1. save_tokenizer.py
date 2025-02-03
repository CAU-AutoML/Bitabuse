# Define tokenizer

from tokenizers import Tokenizer, decoders, models

with open("characters.txt") as f:
    characters = f.read().splitlines()

vocab = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[MASK]": 3,
    "[UNK]": 4,
}

for i, char in enumerate(characters):
    vocab[char] = i + 5

for i in range(200):
    vocab[f"[unused{i}]"] = i + 5 + len(characters)

tokenizer = Tokenizer(
    models.BPE(unk_token="[UNK]", vocab=vocab, merges=[], ignore_merges=True)
)

tokenizer.decoder = decoders.ByteLevel()

# Wrap the tokenizer & save it
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    do_lower_case=False,
    clean_text=False,
    strip_accents=False,
    tokenize_chinese_chars=False,
    model_max_length=512,
)


tokenizer.add_special_tokens(
    {
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
    }
)
tokenizer.save_pretrained("character_bert")

from datasets import load_dataset

ds = load_dataset("AutoML/bitaubse")

test_sents = ds["train"]["text"]

print(len(test_sents))

## Test the tokenizer
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("character_bert")

for i, text in tqdm(enumerate(test_sents), total=len(test_sents)):
    e = tokenizer(text, padding=True, truncation=True)
    # d = tokenizer.decode(e["input_ids"], skip_special_tokens=True) # .decode has a bug
    d = "".join(
        tokenizer.convert_ids_to_tokens(e["input_ids"], skip_special_tokens=True)
    )
    assert text == d, f"\n{text} !=\n{d}"
