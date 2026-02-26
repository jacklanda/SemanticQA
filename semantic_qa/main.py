# -*- coding: utf-8 -*-

import os
import time
import argparse

from rich.console import Console
from transformers import GPT2Tokenizer

from args import parse_arguments
from eval import compute_metric
from utils import (
    load_prompt,
    load_taxonomy,
    load_tokenizer,
    dump_json,
    dump_tsv,
)
from data_utils import (
    get_idiom_data,
    get_collocation_data,
    get_noun_compound_data,
    get_vmwe_data,
    load_data_example,
    postprocess,
    construct_prompt,
    get_sequential_task_data,
)
from model import query_openai, query_claude, query_gemini, AVAILABLE_MODELS


console = Console()


def query_llm(args: argparse.Namespace, prompt: str) -> str:
    """
    Query LLM model with arguments and user prompt

    Args:
    - args: argparse.Namespace
    - prompt: str

    Returns:
    - response: str
    """
    model = args.model
    # TODO: Some parts need to be modified
    if "gpt-3.5-turbo" in model or "gpt-4" in model or "gpt-5" in model or "o3" in model :
        return query_openai(args, prompt, tok, console)
    elif "claude" in model:
        return query_claude(args, prompt, console)
    elif "gemini" in model:
        # return query_gemini(args, prompt, console)
        return query_openai(args, prompt, tok, console)
    else:
        # use openai pathways as default
        return query_openai(args, prompt, tok, console)


def request_llm(args: argparse.Namespace) -> None:
    task = args.task

    # load prompt template
    console.print(f"[bold green]Loading prompt template for {[task]} ...")
    prompt_template = load_prompt(prompt_path=args.prompt_path)
    # console.print(prompt_template)

    # load taxonomy if needed
    taxonomy = None
    if task in ["collocation-categorization", "collocation-scaling", "sequential-task-collocate-extraction-and-judgment"]:
        console.print("[bold green]Loading taxonomy ...")
        taxonomy = load_taxonomy(args.taxonomy_path, args)

    # load data for inference
    preds, golds = [], []
    console.print("[bold green]Loading data ...")
    if task in ["idiom-detection", "idiom-extraction", "idiom-paraphrase"]:
        instances = get_idiom_data(
            data_path=args.input_path,
            task_type=task,
            max_num_limit=args.shot_num,
        )
    elif task in [
        "collocate-retrieval",
        "collocation-categorization",
        "collocation-extraction",
        "collocation-interpretation",
        "collocation-scaling"
    ]:
        instances = get_collocation_data(
            data_path=args.input_path,
            task_type=task,
            max_num_limit=args.max_query,
        )
    elif task in [
        "noun-compound-compositionality",
        "noun-compound-extraction",
        "noun-compound-interpretation",
    ]:
        instances = get_noun_compound_data(
            data_path=args.input_path,
            task_type=task,
            max_num_limit=args.max_query,
        )
    elif task in ["vmwe-extraction"]:
        instances = get_vmwe_data(
            data_path=args.input_path,
            task_type=task,
            max_num_limit=args.max_query,
        )
    elif task in [
        "sequential-task-collocate-extraction-and-judgment", 
        "sequential-task-idiom-extraction-and-judgment",
        "sequential-task-noun-compound-extraction-and-judgment",
        "sequential-task-idiom-extraction-and-interpretation",
        "sequential-task-collocate-extraction-and-interpretation",
        "sequential-task-noun-compound-extraction-and-interpretation"
     ]:
        instances = get_sequential_task_data(
            data_path=args.input_path,
            task_type=task,
            max_num_limit=args.max_query,
        )
    else:
        raise ValueError("Task type is not supported!")
    

    # load examples for demonstration
    examples = None
    if args.shot_num > 0:
        console.print("[bold green]Loading examples for demonstration ...")
        examples = load_data_example(
            data_path=args.example_path,
            task_type=args.task,
            max_num_limit=args.shot_num,
        )

    os.makedirs("results", exist_ok=True)
    json_list_output = list()
    console.rule(title="[bold yellow]Start Benchmarking[/bold yellow]")
    with console.status(
        f"[bold yellow]{[args.model]} Benchmarking {task} ({args.shot_num}-shot) ..."
    ) as _:
        idx = 1
        if args.evaluate:
            metric_sum = 0.0
            extracted_correct = 0.0
            both_correct = 0.0
            idiom_extracted_correct = 0.0
            idiom_both_correct = 0.0
            paraphrase_similarity = 0.0
            cond_paraphrase_similarity = 0.0
            total_paraphrase_similarity = 0.0

        for instance in instances:
            if idx - 1 >= args.max_query:
                break
                
            prompt = construct_prompt(
                args, instance, prompt_template, taxonomy, examples, args.oracle_prompt
            )
            
            response = postprocess(query_llm(args, prompt), task, args)
            
            if response == "":
                continue
            # Display predictive result
            if task == "collocation-identification":
                console.print(
                    f"[bold dark_olive_green3][Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]Gold: [underline][{instance['label']}] ({instance['collocation']})[/underline][/bold purple]"
                )
            elif task == "collocation-extraction":
                if args.evaluate:
                    metric_sum += compute_metric(
                        metric_name="exact-match",
                        pred=response,
                        gold=instance["collocation"],
                    )
                    console.print(
                        f"[bold dark_olive_green3]\[Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]\[Gold] [underline]{instance['collocation']}[/underline][/bold purple]\t[bold orange_red1]\[Exact Match] [underline]{round(metric_sum/idx, 4)}[/underline][/bold orange_red1]"
                    )
                else:
                    console.print(
                        f"[bold dark_olive_green3][Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]Gold: [underline]({instance['collocation']})[/underline][/bold purple]"
                    )
            elif task in ["collocation-categorization", "collocation-scaling"]:
                if args.evaluate:
                    preds.append(response)
                    golds.append(instance["label"])
                    acc, macro_f1, micro_f1, weighted_f1 = compute_metric(
                        metric_name="acc-f1",
                        pred=response,
                        gold=instance["label"],
                        preds=preds,
                        golds=golds,
                    )
                    metric_sum += acc
                    console.print(
                        f"[bold dark_olive_green3]\[Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]\[Gold] [underline]{instance['label']}[/underline][/bold purple]\t[bold orange_red1]\[Accuracy] [underline]{round(metric_sum/idx, 4)}[/underline][/bold orange_red1]\t[bold orange_red1]\[Macro-F1] [underline]{round(macro_f1, 4)}[/underline][/bold orange_red1]\t[bold orange_red1]\[Micro-F1] [underline]{round(micro_f1, 4)}[/underline][/bold orange_red1]\t[bold orange_red1]\[Weighted-F1] [underline]{round(weighted_f1, 4)}[/underline][/bold orange_red1]"
                    )
                else:
                    console.print(
                        f"[bold dark_olive_green3][Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]Gold: [underline]({instance['label']})[/underline][/bold purple]"
                    )
            elif task == "collocation-interpretation":
                if args.evaluate:
                    bert_scores=[]
                    for gold_ref in instance["paraphrases"]:
                        score = compute_metric(
                            metric_name="bert-score",
                            pred=response,
                            gold=gold_ref
                        )
                        bert_scores.append(score)
                    max_bert_score = max(bert_scores)
                    metric_sum += max_bert_score
                    console.print(
                        f"[bold dark_olive_green3]\[Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]\[Gold] [underline]{instance['paraphrases']}[/underline][/bold purple]\t[bold orange_red1]\[BertScore] [underline]{round(metric_sum/idx, 4)}[/underline][/bold orange_red1]"
                    )
                else:
                    console.print(
                        f"[bold dark_olive_green3][Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]Gold: [underline]{instance['context_literal']}[/underline][/bold purple]"
                    )

            elif task == "idiom-detection":
                if args.evaluate:
                    metric_sum += compute_metric(
                        metric_name="mcq-accuracy",
                        pred=response,
                        gold=instance["target"],
                    )
                    console.print(
                        f"[bold dark_olive_green3]\[Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]\[Gold] [underline]{instance['target']}[/underline][/bold purple]\t[bold orange_red1]\[Accuracy] [underline]{round(metric_sum/idx, 4)}[/underline][/bold orange_red1]"
                    )
                else:
                    console.print(
                        f"[bold dark_olive_green3][Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]Gold: [underline]{instance['target']}: {instance[instance['target']]}[/underline][/bold purple]"
                    )
            elif task == "idiom-extraction":
                if args.evaluate:
                    metric_sum += compute_metric(
                        metric_name="exact-match",
                        pred=response,
                        gold=instance["idiom"],
                    )
                    console.print(
                        f"[bold dark_olive_green3]\[Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]\[Gold] [underline]{instance['idiom']}[/underline][/bold purple]\t[bold orange_red1]\[Accuracy (Exact Match)] [underline]{round(metric_sum/idx, 4)}[/underline][/bold orange_red1]"
                    )
                else:
                    console.print(
                        f"[bold dark_olive_green3][Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]Gold: [underline]({instance['idiom']})[/underline][/bold purple]"
                    )
            elif task == "idiom-paraphrase":
                if args.evaluate:
                    metric_sum += compute_metric(
                        metric_name="bert-score",
                        pred=response,
                        gold=instance[
                            # "context_literal"
                            "paraphrase"
                        ],
                    )
                    console.print(
                        f"[bold dark_olive_green3]\[Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]\[Gold] [underline]{instance['paraphrase']}[/underline][/bold purple]\t[bold orange_red1]\[BertScore] [underline]{round(metric_sum/idx, 4)}[/underline][/bold orange_red1]"
                    )
                else:
                    console.print(
                        f"[bold dark_olive_green3][Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]Gold: [underline]{instance['context_literal']}[/underline][/bold purple]"
                    )
            elif task == "noun-compound-compositionality":
                target = instance["target"]
                if args.evaluate:
                    metric_sum += compute_metric(
                        metric_name="mcq-accuracy",
                        pred=response,
                        gold=instance["target"],
                        instance=instance,
                    )
                    console.print(
                        f"[bold dark_olive_green3]\[Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]\[Gold] [underline]{target} ({instance[target]})[/underline][/bold purple]\t[bold orange_red1]\[Accuracy] [underline]{round(metric_sum/idx, 4)}[/underline][/bold orange_red1]"
                    )
                else:
                    console.print(
                        f"[bold dark_olive_green3][Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]Gold: [underline]{target} ({instance[target]})[/underline][/bold purple]"
                    )
            elif task == "noun-compound-interpretation":
                if args.evaluate:
                    metric_sum += compute_metric(
                        metric_name="bert-score",
                        pred=response,
                        gold=instance["references"],
                    )
                    console.print(
                        f"[bold dark_olive_green3]\[Pred] {instance['noun_compound']}: [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]\[Gold] [underline]{instance['references'][0]}[/underline][/bold purple]\t[bold orange_red1]\[BertScore] [underline]{round(metric_sum/idx, 4)}[/underline][/bold orange_red1]"
                    )
                else:
                    console.print(
                        f"[bold dark_olive_green3]\[Pred] {instance['noun_compound']}: [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]Gold: [underline]{instance['references'][0]}[/underline][/bold purple]"
                    )
            elif task == "noun-compound-extraction":
                if args.evaluate:
                    metric_sum += compute_metric(
                        metric_name="exact-match",
                        pred=response,
                        gold=instance["noun_compound"],
                    )
                    console.print(
                        f"[bold dark_olive_green3]\[Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]\[Gold] [underline]{instance['noun_compound']}[/underline][/bold purple]\t[bold orange_red1]\[Exact Match] [underline]{round(metric_sum/idx, 4)}[/underline][/bold orange_red1]"
                    )
                else:
                    console.print(
                        f"[bold dark_olive_green3][Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]Gold: [underline]({instance['noun_compound']})[/underline][/bold purple]"
                    )
            elif task == "vmwe-extraction":
                if args.evaluate:
                    metric_sum += compute_metric(
                        metric_name="exact-match",
                        pred=response,
                        gold=instance["vmwe"],
                    )
                    console.print(
                        f"[bold dark_olive_green3]\[Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]\[Gold] [underline]{instance['vmwe']}[/underline][/bold purple]\t[bold orange_red1]\[Exact Match] [underline]{round(metric_sum/idx, 4)}[/underline][/bold orange_red1]"
                    )
                else:
                    console.print(
                        f"[bold dark_olive_green3][Pred] [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]Gold: [underline]({instance['vmwe']})[/underline][/bold purple]"
                    )

            elif task in ["sequential-task-collocate-extraction-and-judgment"]:
                if args.evaluate:
                    metric = compute_metric(
                        metric_name="collocate-extract-and-category",
                        pred=response,
                        gold={"collocation":instance["collocation"], "label":instance["label"]},
                    )
                    extracted_correct += metric["extracted_correct"]
                    both_correct += metric["both_correct"]
                    cond_rate = (
                        both_correct / extracted_correct if extracted_correct > 0.0 else 0.0
                    )
                    total_rate = both_correct / idx
                    console.print(
                        f"[bold dark_olive_green3]"
                        f"[Pred Collocation] {response['extracted_collocation']}  "
                        f"[Pred Category] {response['category']} "
                        f"[/bold dark_olive_green3]\t"
                        f"[bold purple]"
                        f"[Gold Collocation] {instance['collocation']}  "
                        f"[Gold Category] {instance['label']}"
                        f"[/bold purple]\t"
                        f"[bold orange_red1]"
                        f"\[Cond rate] [underline]{cond_rate:.4f}[/underline]\t"
                        f"\[Total rate] [underline]{total_rate:.4f}[/underline]"
                        f"[/bold orange_red1]"
                    )
                else:
                    console.print(f"[Pred] {response}")

            elif task == "sequential-task-idiom-extraction-and-judgment":
                if args.evaluate:
                    metric = compute_metric(
                        metric_name="idiom-extraction-and-judgment",
                        pred=response,
                        gold={"idiom": instance["idiom"], "target": instance["target"]},
                    )
            
                    idiom_extracted_correct += metric["extracted_correct"]
                    idiom_both_correct += metric["both_correct"]
            
                    cond_rate = (
                        idiom_both_correct / idiom_extracted_correct
                        if idiom_extracted_correct > 0.0 else 0.0
                    )
                    total_rate = idiom_both_correct / idx
            
                    console.print(
                        f"[bold dark_olive_green3]"
                        f"[Pred Idiom] {response['extracted_idiom']}  "
                        f"[Pred Option] {response['option']} "
                        f"[/bold dark_olive_green3]\t"
                        f"[bold purple]"
                        f"[Gold Idiom] {instance['idiom']}  "
                        f"[Gold Target] {instance['target']}"
                        f"[/bold purple]\t"
                        f"[bold orange_red1]"
                        f"\[Cond rate] [underline]{cond_rate:.4f}[/underline]\t"
                        f"\[Total rate] [underline]{total_rate:.4f}[/underline]"
                        f"[/bold orange_red1]"
                    )
            
                else:
                    console.print(f"[Pred] {response}")

            elif task == "sequential-task-noun-compound-extraction-and-judgment":
                if args.evaluate:
                    metric = compute_metric(
                        metric_name="nc-extract-and-judgment",
                        pred=response,
                        gold={
                            "noun_compound": instance["noun_compound"],
                            "target": instance["target"]
                        },
                    )

                    extracted_correct += metric["extracted_correct"]
                    both_correct += metric["both_correct"]

                    cond_rate = (
                        both_correct / extracted_correct if extracted_correct > 0.0 else 0.0
                    )
                    total_rate = both_correct / idx

                    console.print(
                        f"[bold dark_olive_green3]"
                        f"[Pred NC] {response['extracted_collocation']}  "
                        f"[Pred Category] {response['category']} "
                        f"[/bold dark_olive_green3]\t"
                        f"[bold purple]"
                        f"[Gold NC] {instance['noun_compound']}  "
                        f"[Gold Category] {instance['target']}"
                        f"[/bold purple]\t"
                        f"[bold orange_red1]"
                        f"\[Cond rate] [underline]{cond_rate:.4f}[/underline]\t"
                        f"\[Total rate] [underline]{total_rate:.4f}[/underline]"
                        f"[/bold orange_red1]"
                    )
                else:
                    console.print(f"[Pred] {response}")
            
            elif task == "sequential-task-idiom-extraction-and-interpretation":
                if args.evaluate:                
                    metric = compute_metric(
                        metric_name="ie-extraction-and-interpretation",
                        pred=response,
                        gold={
                            "idiom": instance["idiom"],
                            "paraphrase": instance["paraphrase"]
                        },
                    )

                    extracted_correct += metric["extracted_correct"]
                    cond_paraphrase_similarity += metric["cond_paraphrase_similarity"]
                    total_paraphrase_similarity += metric["total_paraphrase_similarity"]

                    cond_rate = (
                        cond_paraphrase_similarity / extracted_correct if extracted_correct > 0.0 else 0.0
                    )
                    comprehensive_rate = cond_paraphrase_similarity / idx
                    total_rate = total_paraphrase_similarity / idx

                    console.print(
                        f"[bold dark_olive_green3]"
                        f"[Pred Idiom] {response['extracted_idiom']}  "
                        f"[Pred Paraphrase] {response['paraphrase']} "
                        f"[/bold dark_olive_green3]\t"
                        f"[bold purple]"
                        f"[Gold Idiom] {instance['idiom']}  "
                        f"[Gold Paraphrase] {instance['paraphrase']}"
                        f"[/bold purple]\t"
                        f"[bold orange_red1]"
                        f"\[Cond rate] [underline]{cond_rate:.4f}[/underline]\t"
                        f"\[Comprehensive rate] [underline]{comprehensive_rate:.4f}[/underline]\t"
                        f"\[Total rate] [underline]{total_rate:.4f}[/underline]"
                        f"[/bold orange_red1]"
                    )
                else:
                    console.print(f"[Pred] {response}")

            elif task == "sequential-task-collocate-extraction-and-interpretation":
                if args.evaluate:
                    metric = compute_metric(
                        metric_name="lc-extraction-and-interpretation",
                        pred=response,
                        gold={
                            "collocation": instance["collocation"],
                            "paraphrases": instance["paraphrases"]
                        },
                    )

                    extracted_correct += metric["extracted_correct"]
                    cond_paraphrase_similarity += metric["cond_paraphrase_similarity"]
                    total_paraphrase_similarity += metric["total_paraphrase_similarity"]
                    cond_rate = (
                        cond_paraphrase_similarity / extracted_correct if extracted_correct > 0.0 else 0.0
                    )
                    comprehensive_rate = cond_paraphrase_similarity / idx
                    total_rate = total_paraphrase_similarity / idx

                    console.print(
                        f"[bold dark_olive_green3]"
                        f"[Pred Collocation] {response['extracted_collocation']}  "
                        f"[Pred Paraphrase] {response['paraphrase']} "
                        f"[/bold dark_olive_green3]\t"
                        f"[bold purple]"
                        f"[Gold Collocation] {instance['collocation']}  "
                        f"[Gold Paraphrase] {instance['paraphrases']}"
                        f"[/bold purple]\t"
                        f"[bold orange_red1]"
                        f"\[Cond rate] [underline]{cond_rate:.4f}[/underline]\t"
                        f"\[Comprehensive rate] [underline]{comprehensive_rate:.4f}[/underline]\t"
                        f"\[Total rate] [underline]{total_rate:.4f}[/underline]"
                        f"[/bold orange_red1]"
                    )
                else:
                    console.print(f"[Pred] {response}")




            console.log(f"Query {idx} completed.")
            idx += 1
            instance["prediction"] = response
            json_list_output.append(instance)
            

        if ".json" in args.output_path:
            dump_json(args.output_path, json_list_output)
        elif ".tsv" in args.output_path:
            dump_tsv(args.output_path, json_list_output)
        console.rule(title="[bold yellow]End Benchmarking[/bold yellow]")

    console.rule(title="[bold yellow]Performance Report[/bold yellow]")
    # TODO: add performance reporter
    console.rule(title="")


if __name__ == "__main__":
    # parsing input arguments
    args = parse_arguments()

    
    if args.model in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-5", "o3"]:# "GPT-5", "o3" 
        tok = load_tokenizer(args.model)
    else:
        tok = GPT2Tokenizer.from_pretrained("gpt2") 

    if args.model in AVAILABLE_MODELS:
        request_llm(args)  # request GPT-3.5-turbo / GPT-4
    else:
        raise ValueError("Model name is not supported!")
