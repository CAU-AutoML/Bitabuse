# Experimental Code for the Paper "BitAbuse: A Dataset of Visually Perturbed Texts for Defending Phishing Attacks"

## 1. Character BERT Training

### 1.1. `save_tokenizer.py`

The tokenizer for the Character BERT model is saved in the "character_bert" folder.

### 1.2. `train_and_eval.py`

We used the Character BERT tokenizer and trained the BERT model from scratch on the bitcore, bitviper, and bitabuse datasets. We also trained models using different dataset split configurations.

## 2. Experiments

### 2.0. Dataset

You can download the datasets from the following link: [Bitabuse](https://huggingface.co/datasets/AutoML/bitaubse), [Bitviper](https://huggingface.co/datasets/AutoML/bitviper), [Bitcore](https://huggingface.co/datasets/AutoML/bitcore).

### 2.1. `infer_ocr_and_simchar.py`

We used an OCR and SimChar Database-based model to restore the perturbed text and saved the results as a CSV file in the "output" folder.

### 2.2. `infer_spellchecker.py`

We used a SpellChecker-based model to restore the perturbed text and saved the results as a CSV file in the "output" folder. Multithreading was used to speed up the process.

### 2.3. `infer_charbert.py`

We used the Character BERT-based model, which was pretrained on the bitcore, bitviper, and bitabuse datasets, to restore the perturbed text. The results were saved in the "output" folder as a CSV file for each dataset split configuration and dataset shuffle seed.

### 2.4. `gpt-4o-mini/`

To use the GPT-4o-mini model, we utilized OpenAI's batch inference API. We saved the batched dataset in the "gpt_batches" folder as a JSONL file and used the API to restore the perturbed text.

### 2.5. `eval.ipynb`

This notebook gathers the results from the OCR and SimChar Database-based model, SpellChecker-based model, Character BERT-based model, and GPT-4o-mini model. It evaluates the performance of each model.