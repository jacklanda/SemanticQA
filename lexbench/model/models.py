# -*- coding: utf-8 -*-
#
# @Author: YangLiu <yangliu.real@gmail.com>
# @Date: 2023/12/27

from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel


AVAILABLE_MODELS = [
    "gpt-4-1106-preview",  # GPT-4 Turbo
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-0314",
    "gpt-3.5-turbo-1106",  # Updated GPT 3.5 Turbo
    "gpt-3.5-turbo",  # Currently points to gpt-3.5-turbo-0613.
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0301",
    "text-davinci-003",
    "claude-instant",
    "claude-instant-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-3-sonnet",
    "claude-3-sonnet-20240229",
    "claude-3-opus",
    "claude-3-opus-20240229",
    "gemini-pro",
    "gemini-1.5-pro-latest",
    "deepseek-chat",
    "lmsys/vicuna-7b-v1.5",  # open model
    "THUDM/chatglm3-6b",  # open model
    "mistralai/Mistral-7B-Instruct-v0.2",  # open model
    "./Mistral-7B-Instruct-v0.2",  # open model
    "NousResearch/Llama-2-7b-chat-hf",  # open model
    "4bit/Llama-2-7b-chat-hf",  # open model
    "01-ai/Yi-6B-Chat",  # open model
    "llama2-13b-hf",  # open model
    "meta-llama/Llama-2-7b-chat-hf",  # open model
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "vicuna-13b-v1.5",  # open model
    "chatglm-2",  # open model
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mixtral-8×7b-instruct-v0.1",  # open model
]


# implement a dataclass to define a list of available models
@dataclass
class Model(BaseModel):
    name: str
    context_size: int
    publish_date: datetime
    description: Optional[str] = None
    organization: Optional[str] = None
