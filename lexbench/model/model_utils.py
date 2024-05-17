# -*- coding: utf-8 -*-
#
# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2024/01/30

import os
import argparse
import numpy as np
from time import sleep
from typing import Any, Dict, List, Optional

import openai
import anthropic
from google import generativeai as genai
from anthropic import HUMAN_PROMPT, AI_PROMPT
from httpx import Client
from rich.console import Console


HTTPX_CLIENT = Client(
    proxies={"https://": "http://127.0.0.1:7890", "http://": "http://127.0.0.1:7890"}
)


def query_openai(
    args: argparse.Namespace, prompt: str, tok: Any, console: Console, **kwargs
) -> str:
    """
    Query OpenAI API.

    Args:
    - args: argparse.Namespace
    - prompt: str

    Returns:
    - response: str
    """
    # setting up OpenAI API key
    openai.api_key = (
        args.api_key if args.api_key is not None else os.environ.get("OPENAI_KEY")
    )

    # setting up OpenAI API base url
    if args.base_url:
        openai.api_base = args.base_url

    # use proxy
    if args.proxy:
        openai.proxy = (
            {"http": args.proxy, "https": args.proxy}
            if args.proxy is not None
            else None
        )

    # truncate for each prompt up to the limitation of `max_tokens`
    model = args.model
    accumulate_count = 0
    if "gpt-4" in model:
        max_context_length = 8192 - 1024 - args.max_tokens
    elif model == "gpt-3.5-turbo":
        max_context_length = 16384 - 1024 - args.max_tokens
    else:
        max_context_length = 8192
    input_ids = tok.encode(prompt)
    prompt = tok.decode(
        input_ids[: max_context_length - args.max_tokens],
    )
    messages = [
        {"role": "user", "content": prompt},
    ]
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                logprobs=True,
                top_logprobs=5,
                **kwargs,
            )
        except Exception as e:
            if accumulate_count > 2:
                console.log("Retrying failed, bypass this example.")
                return ""
            accumulate_count += 1
            max_context_length = max_context_length - 512
            prompt = tok.decode(
                input_ids[: max_context_length - args.max_tokens],
            )  # decreasing prompt length
            messages = [
                {"role": "assistant", "content": prompt},
            ]
            console.log("Retrying:", str(e).split(" (")[0])
            sleep(10)
        else:
            break

    if args.debug:
        top_two_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        for i, logprob in enumerate(top_two_logprobs, start=1):
            pred_detail = f"token: {logprob.token}\tlogprob: {logprob.logprob}\tprob: {np.exp(logprob.logprob) * 100:.2f}%"
            console.print(pred_detail)

    return (
        response.strip()
        if isinstance(response, str)
        else response.choices[0].message.content.strip()
    )


def query_claude(args: argparse.Namespace, prompt: str, console: Console) -> str:
    """
    Query Claude API.

    Args:
    - args: argparse.Namespace
    - prompt: str

    Returns:
    - response: str
    """
    accumulate_count = 0
    max_context_length = 16384 - 1024 - args.max_tokens
    prompt = " ".join(prompt.split(" ")[: int(max_context_length)])
    client = anthropic.Client(
        api_key=args.api_key,
        timeout=1800,
        http_client=HTTPX_CLIENT,
    )
    while True:
        try:
            if "claude-3" in args.model:
                response = (
                    client.messages.create(
                        model=args.model,
                        max_tokens=args.max_tokens,
                        system="",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt,
                                    }
                                ],
                            }
                        ],
                    )
                    .content[0]
                    .text.strip()
                )
            else:
                response = client.completions.create(
                    model=args.model,
                    max_tokens_to_sample=args.max_tokens,
                    prompt=f"{HUMAN_PROMPT}{prompt}{AI_PROMPT}",
                ).completion.strip()
        except Exception as e:
            if accumulate_count > 2:
                console.log("Retrying failed, bypass this instance.")
                return ""
            accumulate_count += 1
            prompt = " ".join(
                prompt.split(" ")[: (int(max_context_length - 512))]
            )  # decreasing prompt length
            console.log(f"Retrying: {e}")
            sleep(10)
        else:
            if any(
                [p in response for p in ["Sorry", "Okay", "Unfortunately", "I don't"]]
            ):
                if accumulate_count > 2:
                    console.log("Retrying failed, return empty literal string.")
                    return ""
                console.log("Detecting rejective response, retry this instance.")
                accumulate_count += 1
                continue
            return response


def query_gemini(
    args: argparse.Namespace,
    prompt: str,
    console: Console,
    ensure_safety_response: bool = True,
) -> str:
    """
    Query Gemini API.

    Args:
    - args: argparse.Namespace
    - prompt: str
    - ensure_safety_response: bool

    Returns:
    - response: str
    """
    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel(args.model)
    max_context_length = 30000 - args.max_tokens
    prompt = " ".join(prompt.split(" ")[: int(max_context_length)])
    safety_settings: Optional[List[Dict[str, str]]] = (None,)
    accumulate_count = 0
    if ensure_safety_response:
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
    while True:
        try:
            responses = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
                safety_settings=safety_settings,
            ).parts
            if len(responses) > 0:
                return responses[0].text.strip()
            else:
                if accumulate_count > 2:
                    return ""
                sleep(3)
                accumulate_count += 1
                continue
        except Exception as e:
            if accumulate_count > 2:
                console.log("Retrying failed, bypass this example.")
                return ""
            accumulate_count += 1
            prompt = " ".join(prompt.split(" ")[: (int(max_context_length - 512))])
            console.log(f"Retrying: {e}")
            sleep(3)
