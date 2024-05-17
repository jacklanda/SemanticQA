# -*- coding: utf-8 -*-
#
# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2024/01/10

import json
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
from evaluate import load
from rich.console import Console
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, util


if __name__ != "__main__":
    ROUGE_SCORER = load("rouge")
    BERT_SCORER = load("bertscore")

    # MODEL_ST = SentenceTransformer(
    # model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    # )


def compute_macro_f1(preds: List[str], golds: List[str]) -> float:
    """
    Compute macro f1 score according to the input `pred_class_map` and `gold_class_map`,
    which include prediction class/gold class (str) to number.
    """
    return f1_score(y_pred=preds, y_true=golds, average="macro")


def compute_micro_f1(preds: List[str], golds: List[str]) -> float:
    """
    Compute micro f1 score dynamically according to the input `pred_class_map` and `gold_class_map`,
    which include prediction class/gold class (str) to number.
    """
    return f1_score(y_pred=preds, y_true=golds, average="micro")


def compute_weighted_f1(preds: List[str], golds: List[str]) -> float:
    """
    Compute weighted f1 score dynamically according to the input `pred_class_map` and `gold_class_map`,
    which include prediction class/gold class (str) to number.
    """
    return f1_score(y_pred=preds, y_true=golds, average="weighted")


def compute_metric(
    metric_name: str,
    pred: str,
    gold: str,
    *args,
    preds: Optional[List[str]] = None,
    golds: Optional[List[str]] = None,
    instance: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> float:
    """
    Compute metric according to the input `metric_name`, `pred` and `gold`.

    Args:
    - metric_name: str, the name of the metric.
    - pred: str, the prediction.
    - gold: str, the ground truth.
    - instance: Dict[str, Any], the instance to be evaluated.

    Returns:
    - float, the metric value.
    """
    # remove trailing spaces for each item
    if isinstance(pred, list):
        pred = [p.strip() for p in pred]
    if isinstance(gold, list):
        gold = [g.strip() for g in gold]
    if isinstance(pred, str):
        pred = pred.strip()
    if isinstance(gold, str):
        gold = gold.strip()

    if metric_name == "bert-score":
        return BERT_SCORER.compute(predictions=[pred], references=[gold], lang="en")[
            "f1"
        ][0]
    if metric_name == "acc":
        return 1.0 if pred == gold else 0.0
    if metric_name == "f1":
        return [
            compute_macro_f1(preds, golds),
            compute_micro_f1(preds, golds),
            compute_weighted_f1(preds, golds),
        ]
    if metric_name == "acc-f1":
        return [
            1.0 if pred == gold else 0.0,
            compute_macro_f1(preds, golds),
            compute_micro_f1(preds, golds),
            compute_weighted_f1(preds, golds),
        ]
    if metric_name == "macro-f1":
        return compute_macro_f1(preds, golds)
    if metric_name == "micro-f1":
        return compute_micro_f1(preds, golds)
    if metric_name == "weighted-f1":
        return compute_weighted_f1(preds, golds)
    if metric_name == "mcq-accuracy":
        if pred == gold:
            return 1.0
        if ":" in pred:
            option_pred = pred.split(":")[0]
            if option_pred == gold:
                return 1.0
        if instance is not None:
            gold_text = instance.get(instance["target"], "")
            if gold_text.lower() == pred.lower():
                return 1.0
        return 0.0
    if metric_name == "exact-match":
        pred, gold = pred.lower(), gold.lower()
        if pred == gold:
            return 1.0
        elif " is " + gold in pred:
            return 1.0
        elif gold + " this " in pred:
            return 1.0
        elif gold + " the " in pred:
            return 1.0
        elif gold + " it " in pred:
            return 1.0
        elif gold + " in " in pred:
            return 1.0
        elif gold + " explanation" in pred:
            return 1.0
        if gold + " is a noun compound" in pred:
            return 1.0
        if gold + " refers to" in pred:
            return 1.0
        return 0.0
    else:
        raise NotImplementedError(f"Metric {metric_name} not implemented!")


def perform_eval(task="vmwe", result_file: Optional[str] = None):
    console = Console()
    if task == "vmwe":
        label2count = dict()
        label2count_macro = dict()
        with open(result_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                result_obj = json.loads(line)
                label = result_obj["label"]
                root_label = label.split(".")[0]
                vmwe = result_obj["vmwe"]
                prediction = result_obj["prediction"]

                if label not in label2count:
                    label2count[label] = {"total": 0, "correct": 0}
                label2count[label]["total"] += 1

                if root_label not in label2count_macro:
                    label2count_macro[root_label] = {"total": 0, "correct": 0}
                label2count_macro[root_label]["total"] += 1

                if vmwe == prediction or "is " + vmwe in prediction:
                    label2count[label]["correct"] += 1
                    label2count_macro[root_label]["correct"] += 1

        for label, count in label2count.items():
            console.print(
                f"Label: {label}, Total: {count['total']}, Correct: {count['correct']}, Accuracy: {round(count['correct']/count['total'], 4) * 100}"
            )
        # display accuracy for each root label
        for label, count in label2count_macro.items():
            console.print(
                f"Label: {label}, Total: {count['total']}, Correct: {count['correct']}, Accuracy: {round(count['correct']/count['total'], 4) * 100}"
            )
    elif task == "nce":
        total, correct = 0, 0
        with open(result_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                result_obj = json.loads(line)
                label = result_obj["noun_compound"]
                prediction = result_obj["prediction"]
                if prediction == label:
                    correct += 1
                total += 1
        console.print(
            f"Total: {total}, Correct: {correct}, Accuracy: {round(correct/total, 4) * 100}"
        )
    elif task == "lcc":
        label2count = dict()
        gold2pred = dict()
        labels = [
            "Magn",
            "AntiMagn",
            "Ver",
            "AntiVer",
            "Bon",
            "AntiBon",
            "Son",
            "Oper1",
        ]
        with open(result_file, "r") as f:
            lines = f.readlines()
            # calc each category total number and prediction distribution in the category
            for line in lines:
                result_obj = json.loads(line)
                label = result_obj["label"]
                prediction = result_obj["prediction"]
                if label not in label2count:
                    label2count[label] = {"total": 0, "correct": 0}
                label2count[label]["total"] += 1
                if prediction == label:
                    label2count[label]["correct"] += 1
                if label not in gold2pred:
                    gold2pred[label] = dict()
                if prediction not in gold2pred[label]:
                    gold2pred[label][prediction] = 0
                gold2pred[label][prediction] += 1
        # display prediction distribution for each label
        for label in labels:
            console.print("  " + label, end="")
        print()
        for label in labels:
            console.print(label[:7], end="\t")
            dist = "\t".join([str(gold2pred[label].get(pred, 0)) for pred in labels])
            console.print(dist)
        print()
        # generate a 2-D matrix from gold2pred
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for i, label in enumerate(labels):
            for j, pred in enumerate(labels):
                matrix[i][j] = gold2pred[label].get(pred, 0)
        matrix_str = "\n".join([",".join([str(e) for e in row]) for row in matrix])
        matrix_str = "[[" + matrix_str.replace("\n", "],\n[") + "]]"
        console.print(matrix_str)
        # display global accuracy
        total, correct = 0, 0
        for label, count in label2count.items():
            total += count["total"]
            correct += count["correct"]
            console.print(
                f"Label: {label}, Total: {count['total']}, Correct: {count['correct']}, Accuracy: {round(count['correct']/count['total'], 4) * 100}"
            )
        print()
        # total accuracy
        console.print(
            f"Total: {total}, Correct: {correct}, Accuracy: {round(correct/total, 4) * 100}"
        )
    elif task in ["iep", "lcp", "ncp"]:
        rouge_scorer = load("rouge")
        bert_scorer = load("bertscore")
        ppl_scorer = load("perplexity", module_type="metric")
        rouge_l, bert_score_f1, ppl = 0.0, 0.0, 0.0
        with open(result_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                result_obj = json.loads(line)
                pred = result_obj["prediction"]
                if "references" in result_obj:
                    gold = result_obj["references"]
                elif "paraphrases" in result_obj:
                    gold = result_obj["paraphrases"]
                if isinstance(gold, list):
                    gold = [gold]
                elif isinstance(gold, str):
                    gold = [gold]
                rouge_l += rouge_scorer.compute(predictions=[pred], references=gold)[
                    "rougeL"
                ]
                bert_score_f1 += bert_scorer.compute(
                    predictions=[pred],
                    references=gold,
                    model_type="roberta-large",
                )["f1"][0]
                ppl += ppl_scorer.compute(predictions=[pred])["mean_perplexity"]
        console.print(
            f"ROUGE-L: {round(rouge_l/len(lines), 4) * 100}\tBERT-Score F1: {round(bert_score_f1/len(lines), 4) * 100}\tPPL: {round(ppl/len(lines), 4)}"
        )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--task", required=True, help="Task to evaluate.")
    parser.add_argument(
        "--result_file_path", required=True, help="Path to result file."
    )
    parsed_args = parser.parse_args()
    perform_eval(task=parsed_args.task, result_file=parsed_args.result_file_path)
