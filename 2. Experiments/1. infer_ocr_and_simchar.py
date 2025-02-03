import time

from datasets import load_dataset
from methods import OCR, Simchar
from misc import ascii_lower
from tqdm import tqdm

DATASET_NAME = "AutoML/bitabuse"
bitabuse = load_dataset(DATASET_NAME, split="train").to_pandas()
bitabuse.set_index("id", inplace=True)

ocr = OCR()
simchar = Simchar()


for i, row in tqdm(bitabuse.iterrows(), total=len(bitabuse)):
    text, label = row["text"], row["label"]

    start = time.time()
    ocr_text = ocr(text)
    ocr_time = time.time() - start
    bitabuse.at[i, "ocr"] = ascii_lower(ocr_text)
    bitabuse.at[i, "ocr_time"] = ocr_time

    start = time.time()
    simchar_text = simchar(text)
    simchar_time = time.time() - start
    bitabuse.at[i, "simchar"] = ascii_lower(simchar_text)
    bitabuse.at[i, "simchar_time"] = simchar_time

bitabuse.to_csv("./outputs/bitabuse_infer_ocr_simchar.csv")
