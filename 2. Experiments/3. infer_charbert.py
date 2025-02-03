import os
import time

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.dataset import train_valid_test_split
from src.utils import set_seed_for_reproducibility

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_seed_for_reproducibility(42)

#####################
SETTINGS = ("bitcore", "AutoML/bitcore", "bitcore")
SPLIT_RATIO = (6, 2, 2)
#####################


def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_text = "".join(tokenizer.convert_ids_to_tokens(predicted_ids[0]))
    return predicted_text


def infer(model_conf, dataset_conf, save_dir_conf, split_ratio):
    ds = load_dataset(dataset_conf, split="train")

    for seed in range(10):
        _, _, ds_test = train_valid_test_split(ds, split_ratio=split_ratio, seed=seed)

        model_id = (
            f"AutoML/CharacterBERT-{model_conf}-{'-'.join(split_ratio)}-seed{seed}"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id).cuda()

        outputs = {"id": [], "charbert": [], "charbert_time": []}
        for example in tqdm(ds_test):
            text = example["text"]
            s = time.time()
            predicted_text = predict(model, tokenizer, text)
            charbert_time = time.time() - s
            outputs["id"].append(example["id"])
            outputs["charbert"].append(predicted_text)
            outputs["charbert_time"].append(charbert_time)

        save_path = f"./outputs/charbert-{save_dir_conf}/{seed}.csv"

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        pd.DataFrame(outputs).to_csv(save_path, index=False)


infer(*SETTINGS, SPLIT_RATIO)
