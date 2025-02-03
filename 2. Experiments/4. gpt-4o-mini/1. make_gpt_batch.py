import json

from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("AutoML/bitabuse", split="train")

prompt = """Restore the In Text to its original Out Text (output only output text):
In Text: {text}
Out Text: """

line_format = {
    "custom_id": "{id}",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": 300,
    },
}

batch_size = 20000
batches = [
    dataset.to_list()[i : i + batch_size] for i in range(0, len(dataset), batch_size)
]


out_path = "./gpt_batches/gpt_batch.jsonl"
for batch_num, batch in enumerate(batches):
    with open(out_path.replace(".jsonl", f"_{batch_num}.jsonl"), "w") as f:
        for id, example in tqdm(enumerate(batch), total=len(batch)):
            line = line_format.copy()
            line["custom_id"] = str(batch_num) + "_" + str(id)
            line["body"]["messages"][0]["content"] = prompt.format(text=example["text"])
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
