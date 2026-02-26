# -*- coding: utf-8 -*-

import json
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
from evaluate import load
from rich.console import Console
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, util


import sys
log = open("eval_output_old_models.log", "a", encoding="utf-8")

def write_log(msg):
    print(msg)
    log.write(msg + "\n")


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
    if metric_name == "collocate-extract-and-category":
        
        pred_extracted_collocation = (pred.get("extracted_collocation") or "").lower().strip()
        gold_collocation = (gold.get("collocation") or "").lower().strip()
        extracted_correct = 1.0 if pred_extracted_collocation == gold_collocation else 0.0

        
        pred_category = (pred.get("category") or "").lower().strip()
        gold_label = (gold.get("label") or "").lower().strip()
        category_correct = 1.0 if pred_category == gold_label else 0.0
         
        both_correct = 1.0 if (extracted_correct and category_correct) else 0.0
        return {
            "extracted_correct": extracted_correct,
            "category_correct": category_correct,
            "both_correct": both_correct
        }
    
    if metric_name == "idiom-extraction-and-judgment":

        pred_extracted_idiom = (pred.get("extracted_idiom") or "").lower().strip()
        gold_idiom = (gold.get("idiom") or "").lower().strip()
        extracted_correct = 1.0 if pred_extracted_idiom == gold_idiom else 0.0

        pred_option = (pred.get("option") or "").strip().upper()
        gold_target = (gold.get("target") or "").strip().upper()
        option_correct = 1.0 if pred_option == gold_target else 0.0

        both_correct = 1.0 if (extracted_correct and option_correct) else 0.0

        return {
            "extracted_correct": extracted_correct,
            "option_correct": option_correct,
            "both_correct": both_correct,
        }
    
    if metric_name == "nc-extract-and-judgment":

        pred_extracted_collocation = (pred.get("extracted_collocation") or "").lower().strip()
        gold_noun_compound = (gold.get("noun_compound") or "").lower().strip()
        extracted_correct = 1.0 if pred_extracted_collocation == gold_noun_compound else 0.0
        
        pred_category = (pred.get("category") or "").lower().strip()
        gold_target = (gold.get("target") or "").lower().strip()
        category_correct = 1.0 if pred_category == gold_target else 0.0

        both_correct = 1.0 if (extracted_correct and category_correct) else 0.0

        return {
            "extracted_correct": extracted_correct,
            "category_correct": category_correct,
            "both_correct": both_correct,
        }


    
    if metric_name == "ie-extraction-and-interpretation":

        pred_extracted_idiom = (pred.get("extracted_idiom") or "").lower().strip()
        gold_idiom = (gold.get("idiom") or "").lower().strip()
        extracted_correct = 1.0 if pred_extracted_idiom == gold_idiom else 0.0

        pred_paraphrase = (pred.get("paraphrase") or "")
        gold_paraphrase = (gold.get("paraphrase") or "")
        if isinstance(pred, list):
            pred_paraphrase = [p.strip() for p in pred_paraphrase]
        if isinstance(gold, list):
            gold_paraphrase = [g.strip() for g in gold_paraphrase]
        if isinstance(pred, str):
            pred_paraphrase = pred_paraphrase.strip()
        if isinstance(gold, str):
            gold_paraphrase = gold_paraphrase.strip()
        
        if extracted_correct == 1.0:
            cond_paraphrase_similarity = (
                BERT_SCORER.compute(predictions=[pred_paraphrase], references=[gold_paraphrase], lang="en")["f1"][0]
            )
        else:
            cond_paraphrase_similarity = 0.0 

        total_paraphrase_similarity = (
            BERT_SCORER.compute(predictions=[pred_paraphrase], references=[gold_paraphrase], lang="en")["f1"][0]
        )

        return {
            "extracted_correct": extracted_correct,
            "cond_paraphrase_similarity": cond_paraphrase_similarity,
            "total_paraphrase_similarity": total_paraphrase_similarity
        }
    
    if metric_name == "lc-extraction-and-interpretation":
        pred_extracted_collocation = (pred.get("extracted_collocation") or "").lower().strip()
        gold_collocation = (gold.get("collocation") or "").lower().strip()
        extracted_correct = 1.0 if pred_extracted_collocation == gold_collocation else 0.0

        pred_paraphrase = (pred.get("paraphrase") or "")
        gold_paraphrase = (gold.get("paraphrases") or "")
        if isinstance(pred, list):
            pred_paraphrase = [p.strip() for p in pred_paraphrase]
        if isinstance(gold, list):
            gold_paraphrase = [g.strip() for g in gold_paraphrase]
        if isinstance(pred, str):
            pred_paraphrase = pred_paraphrase.strip()
        if isinstance(gold, str):
            gold_paraphrase = gold_paraphrase.strip()

        if extracted_correct == 1.0:
            cond_paraphrase_similarity = (
                BERT_SCORER.compute(predictions=[pred_paraphrase], references=[gold_paraphrase], lang="en")["f1"][0]
            )
        else:
            cond_paraphrase_similarity = 0.0 

        total_paraphrase_similarity = (
            BERT_SCORER.compute(predictions=[pred_paraphrase], references=[gold_paraphrase], lang="en")["f1"][0]
        )

        return {
            "extracted_correct": extracted_correct,
            "cond_paraphrase_similarity": cond_paraphrase_similarity,
            "total_paraphrase_similarity": total_paraphrase_similarity
        }


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
        with open(result_file, "r", encoding="utf-8") as f:
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
        from evaluate import load
        import json
        import sacrebleu
        from sacrebleu.metrics import BLEU

        rouge_scorer = load("rouge")
        bert_scorer = load("bertscore")
        meteor_scorer = load("meteor")
        # bleu_scorer = load("bleu")

        predictions = []
        all_references = []
        meteor_references = []  

        with open(result_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                result_obj = json.loads(line)
                pred = result_obj["prediction"]
                predictions.append(pred)

                if "references" in result_obj:
                    gold = result_obj["references"]
                elif "paraphrases" in result_obj:
                    gold = result_obj["paraphrases"]
                elif "paraphrase" in result_obj:
                    gold = result_obj["paraphrase"]

                if isinstance(gold, list):
                    gold_refs = gold
                elif isinstance(gold, str):
                    gold_refs = [gold]
                else:
                    gold_refs = [str(gold)]

                all_references.append(gold_refs)
                
                meteor_references.append([ref for ref in gold_refs])

        
        rouge_results = rouge_scorer.compute(
            predictions=predictions,
            references=all_references,
            use_aggregator=True
        )
        rouge_l = rouge_results["rougeL"] * 100

        flat_preds = []
        flat_refs = []
        group_sizes = []
        
        for pred, refs in zip(predictions, all_references):
            flat_preds.extend([pred] * len(refs))
            flat_refs.extend(refs)
            group_sizes.append(len(refs))
        
        bert_results = bert_scorer.compute(
            predictions=flat_preds,
            references=flat_refs,
            model_type="roberta-large",
            lang="en"
        )
        
        f1_scores = bert_results["f1"]
        
        idx = 0
        bert_f1_list = []
        for size in group_sizes:
            bert_f1_list.append(max(f1_scores[idx:idx+size]))
            idx += size
        
        bert_score_f1 = sum(bert_f1_list) / len(bert_f1_list) * 100


        meteor_results = meteor_scorer.compute(
            predictions=predictions,
            references=meteor_references
        )
        meteor_score = meteor_results["meteor"] * 100

        
        min_refs = min(len(refs) for refs in all_references)
        references_transposed = [[] for _ in range(min_refs)]
    
        for refs in all_references:
            for i in range(min_refs):
                references_transposed[i].append(refs[i])
    

        bleu1 = BLEU(max_ngram_order=1, effective_order=True) \
            .corpus_score(predictions, references_transposed).score

        bleu2 = BLEU(max_ngram_order=2, effective_order=True) \
            .corpus_score(predictions, references_transposed).score

        bleu3 = BLEU(max_ngram_order=3, effective_order=True) \
            .corpus_score(predictions, references_transposed).score

        bleu4 = BLEU(max_ngram_order=4, effective_order=True) \
            .corpus_score(predictions, references_transposed).score

        write_log(
            f"{parsed_args.result_file_path}\n"
            f"ROUGE-L: {rouge_l:.2f}\tBERT-Score F1: {bert_score_f1:.2f}\tMeteor: {meteor_score:.2f} \n"
            f"BLEU-1: {bleu1:.2f}\tBLEU-2: {bleu2:.2f}\t"
            f"BLEU-3: {bleu3:.2f}\tBLEU-4: {bleu4:.2f}"
        )

  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--task", required=True, help="Task to evaluate.")
    parser.add_argument(
        "--result_file_path", required=True, help="Path to result file."
    )
    parsed_args = parser.parse_args()
    perform_eval(task=parsed_args.task, result_file=parsed_args.result_file_path)

