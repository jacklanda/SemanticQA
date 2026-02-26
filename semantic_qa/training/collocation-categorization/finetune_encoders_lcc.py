# -*- coding: utf-8 -*-

import argparse
from typing import List, Optional

import torch
import evaluate
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class LCCDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_lcc_split(split_path: str, conditioning: bool = False):
    texts = []
    collocations = []
    label_ids = []
    with open(split_path, "r") as f:
        for line in f:
            text, collocation, label_id = line.split("\t")
            if conditioning:
                text = f"{text};{collocation}"
            texts.append(text)
            collocations.append(collocation)
            label_ids.append(int(label_id))

    return texts, collocations, label_ids


def compute_metrics(
    eval_preds, eval_golds: Optional[List[str]] = None, do_predict: bool = False
):
    metric = evaluate.load(
        # "accuracy/accuracy.py", "f1/f1.py", "precision/precision.py", "recall/recall.py"  # FIXME: error in loading
        "accuracy/accuracy.py"
    )
    if not do_predict:
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
    elif do_predict and eval_golds is not None:
        predictions = eval_preds
        labels = eval_golds
    return metric.compute(predictions=predictions, references=labels)


def construct_labelinfo(label_map_path: str):
    label2id = dict()
    with open(label_map_path, "r") as f:
        for line in f:
            label, label_id = line.split("\t")
            label2id[label] = int(label_id)

    return label2id, {v: k for k, v in label2id.items()}, [label for label in label2id]


# Main logic

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="The model name in the experiment.",
    choices=[
        "bert-base-uncased",
        "bert-large-uncased",
    ],
)
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="The directory of the data.",
)
parser.add_argument(
    "--do_conditioning",
    action="store_true",
    help="Whether to condition on collocations.",
)
parser.add_argument(
    "--eval_on_test",
    action="store_true",
    help="Whether to evaluate on the test set.",
)
args = parser.parse_args()

label2id, id2label, label_list = construct_labelinfo(f"{args.data_dir}/labelmap.tsv")

# do not condition on collocations
train_texts, train_collocations, train_label_ids = read_lcc_split(
    f"{args.data_dir}/train_ft.tsv", conditioning=args.do_conditioning
)
val_texts, val_collocations, val_label_ids = read_lcc_split(
    f"{args.data_dir}/valid_ft.tsv", conditioning=args.do_conditioning
)
test_texts, test_collocations, test_label_ids = read_lcc_split(
    f"{args.data_dir}/test_ft.tsv", conditioning=args.do_conditioning
)

if args.eval_on_test:
    val_texts = test_texts
    val_collocations = test_collocations
    val_label_ids = test_label_ids

tokenizer = AutoTokenizer.from_pretrained(args.model)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

train_dataset = LCCDataset(train_encodings, train_label_ids)
val_dataset = LCCDataset(val_encodings, val_label_ids)
test_dataset = LCCDataset(test_encodings, test_label_ids)

config = AutoConfig.from_pretrained(
    args.model,
    num_labels=len(label2id),
    finetuning_task="text-classification",
    trust_remote_code=True,
    problem_type="single_label_classification",
)

training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    do_predict=True,
    output_dir="./results",  # output directory
    num_train_epochs=5,  # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    learning_rate=4e-5,
    warmup_ratio=0.06,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    fp16=False,
    logging_dir="./logs",  # directory for storing logs
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=0,
    evaluation_strategy="epoch",
    # evaluation_strategy="steps",
    # eval_accumulation_steps=10,
)

print(training_args)

model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)
model.config.label2id = label2id
model.config.id2label = id2label

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print(f"Start training [{args.model.split('/')[-1].strip()}] ...")
trainer.train()

print(f"Evaluating fine-tuned [{args.model.split('/')[-1].strip()}] on test set ...")
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
metrics = compute_metrics(preds, eval_golds=test_label_ids, do_predict=True)
print(metrics)
