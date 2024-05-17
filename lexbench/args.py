# -*- coding: utf-8 -*-
#
# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2023/10/31

import argparse

import torch


def parse_arguments(desc: str = "Query Large Language Models"):
    """
    Parse arguments for query large language models.
    :param desc: description of the program.
    :return: parsed arguments.
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The task name in the experiment.",
        choices=[
            "idiom-detection",
            "idiom-extraction",
            "idiom-paraphrase",
            "collocate-retrieval",
            "collocation-categorization",
            "collocation-extraction",
            "collocation-interpretation",
            "collocation-identification",
            "noun-compound-compositionality",
            "noun-compound-extraction",
            "noun-compound-interpretation",
            "vmwe-extraction",
        ],
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="Specify the authorized key of OpenAI.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        required=False,
        help="Specify the base URL of OpenAI.",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Whether use proxy for request OpenAI's API.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specify the model name allowed in OpenAI api list.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="Prompt file path should be specified with vanilla text file format.",
    )
    parser.add_argument(
        "--example_path",
        type=str,
        required=False,
        help="Available prompting example file path.",
    )
    parser.add_argument(
        "--taxonomy_path",
        type=str,
        required=False,
        help="Taxonomy file path should be specified with vanilla text file format.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input file path should be specified with jsonl format.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        default="predictions.json",
        help="Output file path should be specified with jsonl format.",
    )
    parser.add_argument(
        "--max_query",
        type=int,
        default=100,
        help="Maximum allowed number of requests to OpenAI GPT.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum allowed number for new generated tokens for each request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0,
        help="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0,
        help="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        required=False,
        help="Perform dynamic evaluation on the generated results.",
    )
    parser.add_argument(
        "--oracle_prompt",
        action="store_true",
        required=False,
        help="Specify whether conduct the experiment with oracle prompt.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Specify whether conduct the experiment in debug mode.",
    )
    parser.add_argument(
        "--shot_num",
        type=int,
        required=True,
        help="Available prompting example number for demonstration.",
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args
