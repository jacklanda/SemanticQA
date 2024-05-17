# -*- coding: utf-8 -*-
#
# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2023/11/09

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
    if "gpt-3.5-turbo" in model or "gpt-4" in model:
        return query_openai(args, prompt, tok, console)
    elif "claude" in model:
        return query_claude(args, prompt, console)
    elif "gemini" in model:
        return query_gemini(args, prompt, console)
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
    if task in ["collocation-categorization"]:
        console.print("[bold green]Loading taxonomy ...")
        taxonomy = load_taxonomy(args.taxonomy_path, args)
        # console.print(taxonomy)

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
    else:
        raise ValueError("Task type is not supported!")
    # console.print_json(data=instances)
    # exit()

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
        for instance in instances:
            if idx - 1 >= args.max_query:
                break
            prompt = construct_prompt(
                args, instance, prompt_template, taxonomy, examples, args.oracle_prompt
            )
            # console.print(prompt)
            # exit()
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
            elif task == "collocation-categorization":
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

    if args.model in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]:
        tok = load_tokenizer(args.model)
    else:
        # load GPT-2 Tokenizer as default
        tok = GPT2Tokenizer.from_pretrained("gpt2")

    if args.model in AVAILABLE_MODELS:
        request_llm(args)  # request GPT-3.5-turbo / GPT-4
    else:
        raise ValueError("Model name is not supported!")
