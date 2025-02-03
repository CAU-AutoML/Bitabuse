import ssl

import numpy as np
from misc import get_label_words_and_idx
from OpenAttack.metric import BLEU, JaccardWord
from OpenAttack.text_process.tokenizer import PunctTokenizer
from transformers import EvalPrediction, PreTrainedTokenizerBase

# setting openattack metrics
ssl._create_default_https_context = ssl._create_unverified_context

punct_tokenizer = PunctTokenizer()
bleu = BLEU(punct_tokenizer)
jaccard_word = JaccardWord(punct_tokenizer)


def word_accuracy(pred, label, input, debug=False):
    label_words_and_idx = get_label_words_and_idx(label)
    label_words = [word for word, _, _ in label_words_and_idx]

    pred_words = [pred[start:end] for _, start, end in label_words_and_idx]
    input_words = [input[start:end] for _, start, end in label_words_and_idx]

    vp_words = [  # [[pred_word, label_word], ...]
        [p, l] for p, l, i in zip(pred_words, label_words, input_words) if l != i
    ]
    if debug and len(vp_words) == 0:
        print("input:", input)
        print("label:", label)
        print("pred:", pred)
    correct_words = [[p, l] for p, l in vp_words if p == l]

    # if there is no vp word, then warning
    if len(vp_words) == 0:
        print(
            "WARNING: There is no vp word in the input text. Word accuracy is considered to be 1.0."
        )
        print("\tinput:", input)
        print("\tinput_words:", input_words)
    correct, total = len(correct_words), len(vp_words)
    return correct / total if total > 0 else 1.0


class MetricsForCharacterBERT:

    def __init__(self, tokenizer: PreTrainedTokenizerBase, debug=False):
        self.word_accs = []
        self.jaccard_words = []
        self.bleus = []
        self.tokenizer = tokenizer
        self.debug = debug

    def __call__(self, eval_pred: EvalPrediction, compute_result=False):
        pred_ids = eval_pred.predictions.argmax(axis=2).detach().cpu().numpy()
        label_ids = eval_pred.label_ids.detach().cpu().numpy()
        input_ids = eval_pred.inputs["input_ids"].detach().cpu().numpy()

        # Replace -100 ([PAD]) in the labels as we can't decode them.
        pred_ids = np.where(pred_ids != -100, pred_ids, self.tokenizer.pad_token_id)
        label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        input_ids = np.where(input_ids != -100, input_ids, self.tokenizer.pad_token_id)

        pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_texts = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        for i, (pred_text, label_text) in enumerate(zip(pred_texts, label_texts)):
            pred_texts[i] = pred_text[: len(label_text)]

        self.word_accs.extend(
            [
                word_accuracy(pred, label, input, self.debug)
                for pred, label, input in zip(pred_texts, label_texts, input_texts)
            ]
        )

        self.jaccard_words.extend(
            [
                jaccard_word.calc_score(pred, label)
                for pred, label in zip(pred_texts, label_texts)
            ]
        )

        self.bleus.extend(
            [
                bleu.calc_score(label, pred)
                for pred, label in zip(pred_texts, label_texts)
            ]
        )

        if compute_result:
            mean_word_acc = np.mean(self.word_accs)
            mean_jaccard_word = np.mean(self.jaccard_words)
            mean_bleu = np.mean(self.bleus)
            self.word_accs = []
            self.jaccard_words = []
            self.bleus = []
            return {
                "word_accuracy": mean_word_acc,
                "jaccard_word": mean_jaccard_word,
                "bleu": mean_bleu,
            }
