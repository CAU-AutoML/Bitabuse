import gc
import os
import time
from typing import Tuple

import torch
from datasets import load_dataset
from metrics import MetricsForCharacterBERT, word_accuracy
from misc import (
    preprocess_bitcoinabuse_dataset,
    set_seed_for_reproducibility,
    train_valid_test_split,
)
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

import wandb

#########################################################################
# DEBUG = True
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 5e-5
TOKENIZER_PATH = "./character_bert"
DATASET_NAME = "AutoML/bitabuse"
NUM_EPOCHS = 20
MODEL_GROUP_NAME = DATASET_NAME.split("/")[1]
SPLITS = [
    (1, 20, 79),
    (5, 20, 75),
    (10, 20, 70),
    (20, 20, 60),
]
#########################################################################


def train(
    out_dir: str,
    tokenizer_path: str,
    hub_model_id: str,
    hub_dataset_id: str,
    hf_username: str,
    wandb_project: str,
    dataset_shuffle_seed: int,
    dataset_split_ratio: Tuple[int, int, int],
    training_args: TrainingArguments,
    checkpoint_dir=None,
    debug=False,
):
    run = wandb.init(
        project=wandb_project,
        name=f"seed{dataset_shuffle_seed}-{time.time()}",
        reinit=True,
    )

    gc.collect()
    torch.cuda.empty_cache()

    resume_from_checkpoint = checkpoint_dir is not None

    model_repo = hf_username + "/" + hub_model_id

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer.push_to_hub(model_repo, private=True)

    # Preprocess dataset
    ds = load_dataset(hub_dataset_id, split="train")
    ds = preprocess_bitcoinabuse_dataset(ds, tokenizer, tokenizer.model_max_length, 32)
    ds_train, ds_valid, ds_test = train_valid_test_split(
        ds, dataset_split_ratio, dataset_shuffle_seed
    )
    print("dataset stats:")
    print("\ttrain:", ds_train)
    print("\tvalid:", ds_valid)
    print("\ttest:", ds_test)

    # Load model
    if resume_from_checkpoint:
        model = BertForMaskedLM.from_pretrained(checkpoint_dir)
    else:
        model_config = BertConfig(
            max_position_embeddings=tokenizer.model_max_length,
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
        model = BertForMaskedLM(model_config)

    # Training
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model,
        training_args,
        compute_metrics=MetricsForCharacterBERT(tokenizer, debug),
        train_dataset=ds_train if not debug else ds_train.shard(10000, 1),
        eval_dataset=ds_valid if not debug else ds_valid.shard(10000, 1),
        data_collator=data_collator,
    )

    print("output_dir:", out_dir)
    print("train device:", model.device)
    print("trainer:", training_args.device)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    gc.collect()
    torch.cuda.empty_cache()

    test_result = trainer.evaluate(
        ds_test if not debug else ds_test.shard(10000, 1),
        metric_key_prefix="test",
    )

    print("test_result:", test_result)

    trainer.push_to_hub()
    run.finish()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    set_seed_for_reproducibility(42)

    if not os.path.exists(MODEL_GROUP_NAME):
        os.makedirs(MODEL_GROUP_NAME)

    for seed in range(1, 10):
        for split in SPLITS:
            MODEL_NAME = (
                f"{MODEL_GROUP_NAME}-{split[0]}-{split[1]}-{split[2]}-seed{seed}"
            )
            OUT_DIR = os.path.join(MODEL_GROUP_NAME, MODEL_NAME)

            TRAINING_ARGS = TrainingArguments(
                output_dir=OUT_DIR,
                hub_model_id=MODEL_NAME,
                num_train_epochs=NUM_EPOCHS if not locals().get("DEBUG", False) else 1,
                eval_strategy="epoch",
                per_device_train_batch_size=TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=EVAL_BATCH_SIZE,
                batch_eval_metrics=True,
                learning_rate=LEARNING_RATE,
                include_inputs_for_metrics=True,
                report_to="wandb",
                logging_strategy="epoch",
                log_level="warning",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
                push_to_hub=True,
                hub_private_repo=True,
            )

            train(
                out_dir=OUT_DIR,
                tokenizer_path=TOKENIZER_PATH,
                hub_model_id=MODEL_NAME,
                hub_dataset_id=DATASET_NAME,
                hf_username="lhy",
                wandb_project=MODEL_NAME.split("-seed")[0],
                dataset_shuffle_seed=seed,
                dataset_split_ratio=split,
                training_args=TRAINING_ARGS,
                checkpoint_dir=None,
                debug=locals().get("DEBUG", False),
            )
