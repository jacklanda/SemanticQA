# -*- coding: utf-8 -*-
#
# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2024/01/07


# This script is to handle all the tasks that are relevant with
# data processing, data preparation in our paper: ``Benchmarking LLMs for Collocation Understanding''.

import os
import re
import ast
import json
import shutil
import random
import pandas as pd
from pprint import pprint
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# all exps should keep this seed for reproducibility
random.seed(42)


class IdiomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx])
        label = torch.tensor(self.labels[idx])

        return data, label


def prepare_data_idiom_extraction(
    data_path: str = "dataset/train_english.tsv",
    dump_data_path: Optional[str] = None,
    only_valid_example: bool = True,
    dedup_by_idiom: bool = True,
    max_data_limit: Optional[int] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Prepare the idiom data for extraction evaluation.

    Args:
    - data_path: the path of the data file
    - dump_data_path: the path of the prepared data file
    - only_valid_example: whether to only keep the valid examples
    - dedup_by_idiom: whether to deduplicate the data by idiom
    - max_data_limit: the max data limit
    - verbose: whether to print the statistics

    Returns:
    - output_data: the prepared data
    """

    # read the data
    with open(data_path, "r") as f:
        lines = f.readlines()

    sentences = []
    current_sentence = []
    current_phrase = []
    phrase_set = set()
    num_w_phrase = 0
    num_wo_phrase = 0

    for line in lines:
        if line.strip() == "":
            if current_sentence:
                sentences.append((current_sentence, current_phrase))
                current_sentence = []
                current_phrase = []
        else:
            word, label = line.split("\t")
            word = word.strip()
            label = label.strip()
            current_sentence.append(word)
            # if all the labels are "O", then skip this instance
            if label.startswith("B-"):
                current_phrase.append(word)
            elif label.startswith("I-"):
                current_phrase.append(word)

    # Add the last sentence if not empty
    if current_sentence:
        sentences.append((current_sentence, current_phrase))

    if dedup_by_idiom:
        dedup_sentences = []
        for sentence, phrase in sentences:
            if phrase:
                phrase_span = " ".join(phrase)
                if phrase_span not in phrase_set:
                    dedup_sentences.append((sentence, phrase))
                    phrase_set.add(phrase_span)
            else:
                dedup_sentences.append((sentence, phrase))
        sentences = dedup_sentences

    # Create the output
    output_data = []
    for sentence, phrase in sentences:
        sentence = " ".join(sentence)
        if phrase:
            num_w_phrase += 1
            phrase_span = " ".join(phrase)
        else:
            num_wo_phrase += 1
            continue
        output_data.append(f"{sentence}\t{phrase_span}\n")

    if max_data_limit:
        output_data = random.sample(output_data, max_data_limit)

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        with open(dump_data_path, "w") as f:
            f.write(f"Context\tIdiom\n")
            f.writelines(output_data)

    if verbose:
        print(
            f"Total instances: {num_w_phrase + num_wo_phrase}\nInstances w/ phrase: {num_w_phrase}\nInstances w/o phrase: {num_wo_phrase}\n"
        )

    return output_data


def prepare_data_idiom_detection(
    refer_data_path: str,
    idiom_data_path: str,
    dump_data_path: str = "dataset/idiom_detection/prepared/idiom_detection_prepared.tsv",
    dedup_by_idiom: bool = True,
    verbose: bool = True,
) -> List[str]:
    """
    This function is to prepare the idiom data for evaluation.

    Args:
    - refer_data_path: the path of the refer data file
    - idiom_data_path: the path of the idiom data file
    - dump_data_path: the path of the prepared data file
    - dedup_by_idiom: whether to deduplicate the data by idiom
    - verbose: whether to print the statistics

    Returns:
    - data_examples: the prepared data
    """

    data_examples = []

    # read the data
    refer_df = pd.read_csv(refer_data_path)
    idiom_df = pd.read_csv(idiom_data_path)

    # get a dict include the column "sentence1" and "sentence2"
    refer_dict = {
        item["sentence1"]: item["sentence2"]
        for item in refer_df.to_dict(orient="records")
    }

    idiom_df["idiom"] = idiom_df["sentence1"].apply(lambda x: refer_dict.get(x, ""))

    # filter all rows that have the 3 times repeats value of "sentence1" in `idiom_df`
    sent2freq = dict()
    for item in idiom_df["sentence1"].tolist():
        sent2freq[item] = sent2freq.get(item, 0) + 1
    idiom_df["frequency"] = idiom_df["sentence1"].apply(lambda x: sent2freq[x])

    # get the row of frequency == 4
    idiom_df_four = idiom_df[idiom_df["frequency"] == 4]

    # iterate each 4 rows as a group in `idiom_df_four`
    for i in range(0, len(idiom_df_four), 4):
        # assign A, B, C, D to each row in this group
        data_example = {
            "id": str(i // 4).zfill(3),
            "context": idiom_df_four.iloc[i, 1].replace("\t", " "),
            "idiom": idiom_df_four.iloc[i, 3].replace("\t", " "),
        }
        letter_mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        for j in range(4):
            letter = letter_mapping[j]
            data_example[letter] = idiom_df_four.iloc[i + j, 2]

            if idiom_df_four.iloc[i + j, 0] == "1":
                data_example["target"] = letter

        data_examples.append(data_example)

    # deduplicate `data_example` by field "idiom" for each item
    if dedup_by_idiom:
        idx = 0
        dedup_data_examples = []
        dedup_idiom_set = set()
        for item in data_examples:
            dedup_idiom_set.add(item["idiom"] + item["target"])
        for item in data_examples:
            if item["idiom"] + item["target"] in dedup_idiom_set:
                item["id"] = str(idx).zfill(3)
                dedup_data_examples.append(item)
                dedup_idiom_set.remove(item["idiom"] + item["target"])
                idx += 1
        data_examples = dedup_data_examples

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        with open(dump_data_path, "w") as f:
            f.write(f"id\tcontext\tidiom\tA\tB\tC\tD\ttarget\n")
            for item in data_examples:
                f.write(
                    f"{item['id']}\t{item['context']}\t{item['idiom']}\t{item['A']}\t{item['B']}\t{item['C']}\t{item['D']}\t{item['target']}\n"
                )

    if verbose:
        # rich.print_json(data=data_examples)
        print("Total instances:", len(data_examples))

    return data_examples


def prepare_data_idiom_paraphrase_1(
    idiom_data_path: str,
    dump_data_path: str,
    dedup_by_idiom: bool = True,
    verbose: bool = True,
) -> List[str]:
    """
    Prepare the idiom data for paraphrase evaluation.

    Args:
    - idiom_data_path: the path of the idiom data file
    - dump_data_path: the path of the prepared data file
    - dedup_by_idiom: whether to deduplicate the data by idiom
    - verbose: whether to print the statistics
    """
    data_examples = []
    idiom_df = pd.read_csv(idiom_data_path)

    for i in range(len(idiom_df)):
        data_example = {
            "id": str(i).zfill(4),
            "idiom": idiom_df.iloc[i, 0],
            "paraphrase": idiom_df.iloc[i, 1],
            "context_idiomatic": idiom_df.iloc[i, 2],
            "context_literal": idiom_df.iloc[i, 3],
        }
        data_examples.append(data_example)

    if dedup_by_idiom:
        idx = 0
        dedup_data_examples = []
        dedup_idiom_set = set()
        for item in data_examples:
            dedup_idiom_set.add(item["idiom"])
        for item in data_examples:
            if item["idiom"] in dedup_idiom_set:
                item["id"] = str(idx).zfill(3)
                dedup_data_examples.append(item)
                dedup_idiom_set.remove(item["idiom"])
                idx += 1
        data_examples = dedup_data_examples

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        with open(dump_data_path, "w") as f:
            f.write(f"id\tidiom\tparaphrase\tcontext_idiomatic\tcontext_literal\n")
            for item in data_examples:
                f.write(
                    f"{item['id']}\t{item['idiom']}\t{item['paraphrase']}\t{item['context_idiomatic']}\t{item['context_literal']}\n"
                )
    if verbose:
        print("Total instances:", len(data_examples))

    return data_examples


def prepare_data_idiom_paraphrase_2(
    idiom_data_path: str,
    idiom_data_ref_path: str,
    dump_data_path: str,
    dedup_by_idiom: bool = True,
    verbose: bool = True,
) -> List[str]:
    """
    Prepare the idiom data for paraphrase evaluation.

    Args:
    - idiom_data_path: the path of the idiom data file
    - idiom_data_ref_path: the path of the idiom data reference file
    - dump_data_path: the path of the prepared data file
    - dedup_by_idiom: whether to deduplicate the data by idiom
    - verbose: whether to print the statistics
    """
    data_examples = []

    # build reference set of idioms
    idiom2instance = dict()
    with open(idiom_data_ref_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        for line in lines:
            idx, idiom, paraphrase, context_idiomatic, context_literal = line.split(
                "\t"
            )
            data_example = {
                "id": idx,
                "idiom": idiom,
                "paraphrase": paraphrase,
                "context_idiomatic": context_idiomatic,
                "context_literal": context_literal,
            }
            idiom2instance[idiom] = data_example

    with open(idiom_data_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        for line in lines:
            json_obj = json.loads(line)
            idiom, paraphrase, context_idiomatic, context_literal = (
                json_obj["idiom"].strip(),
                json_obj["meaning"].strip(),
                json_obj["narrative"].replace("<b>", "").replace("</b>", "").strip(),
                json_obj[json_obj["correctanswer"].strip()].strip(),
            )
            if idiom not in context_idiomatic:
                continue
            if idiom not in idiom2instance:
                data_example = {
                    "id": str(len(data_examples)).zfill(3),
                    "idiom": idiom,
                    "paraphrase": paraphrase,
                    "context_idiomatic": context_idiomatic,
                    "context_literal": context_literal,
                }
                data_examples.append(data_example)

    if dedup_by_idiom:
        idx = 0
        dedup_data_examples = []
        dedup_idiom_set = set()
        for item in data_examples:
            dedup_idiom_set.add(item["idiom"])
        for item in data_examples:
            if item["idiom"] in dedup_idiom_set:
                item["id"] = str(idx).zfill(3)
                dedup_data_examples.append(item)
                dedup_idiom_set.remove(item["idiom"])
                idx += 1
        data_examples = dedup_data_examples

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        with open(dump_data_path, "w") as f:
            f.write(f"id\tidiom\tparaphrase\tcontext_idiomatic\tcontext_literal\n")
            for item in data_examples:
                f.write(
                    f"{item['id']}\t{item['idiom']}\t{item['paraphrase']}\t{item['context_idiomatic']}\t{item['context_literal']}\n"
                )
    if verbose:
        print("Total instances:", len(data_examples))

    return data_examples


def sample_data_collocate_retrieval(
    data_path: str = "dataset/collocate_retrieval/Collocations_en.csv",
    dump_data_path: Optional[str] = None,
    category_num: int = 8,
    max_instance_num_per_category: int = 30,
    min_context_size: int = 16,
    max_context_size: int = 64,
    dedup: bool = True,
) -> None:
    """
    Sample the collocate retrieval data for evaluation.

    Args:
    - data_path: the path of the data file
    - dump_data_path: the path of the prepared data file
    - category_num: the number of categories to sample
    - max_instance_num_per_category: the max instance number per category
    - min_context_size: the min context size
    - max_context_size: the max context size
    - dedup: whether to deduplicate the data by the whole collocation

    Returns:
    - None
    """
    if category_num == 8:
        lf_category = [
            "Magn",
            "AntiMagn",
            "Ver",
            "AntiVer",
            "Bon",
            "AntiBon",
            "Son",
            "Oper1",
        ]
    else:
        raise NotImplementedError(f"category_num={category_num} is not implemented.")

    # Step 0. load all instances from the original file
    df_all = pd.read_csv(data_path, sep="\t")

    # Step 1. deduplicate by the 2nd and 5th value of column (collocation = base ⊕ collocate), if needed
    if dedup:
        # df_all = df_all.drop_duplicates(subset=[df_all.columns[1]])
        df_all = df_all.drop_duplicates(subset=[df_all.columns[1], df_all.columns[4]])

    # Step 2. filter out all the rows that 11th column's value word size is not in the interval [16, 64]
    df_all = df_all[
        df_all[df_all.columns[10]].apply(
            lambda x: min_context_size <= len(x.split()) <= max_context_size
        )
    ]

    # Step 3. random select 30 examples for each group which is clustered by the 6th column, from the `lf_category`
    df_sample = pd.DataFrame()
    for category in lf_category:
        df_category = df_all[df_all[df_all.columns[5]] == category]
        df_category_sample = df_category.sample(
            n=max_instance_num_per_category,
            random_state=42,
            # replace=True,
        )
        df_sample = df_sample._append(df_category_sample)

    # Step 4. dump the sampled data to the `dump_data_path`
    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        df_sample.to_csv(dump_data_path, sep="\t", index=False)


def prepare_data_collocate_retrieval(
    data_path: str = "dataset/collocate_retrieval/Collocations_en.csv",
    dump_data_path: Optional[str] = None,
    base_word_num: Optional[int] = None,
    collocate_word_num: Optional[int] = None,
    max_data_limit: int = 320,
    max_instance_num_per_category: int = 40,
    mask_collocate: bool = True,
    verbose: bool = True,
) -> List[str]:
    """
    Prepare the collocate retrieval data for evaluation.

    Args:
    - data_path: the path of the data file
    - dump_data_path: the path of the prepared data file
    - base_word_num: the number of words in the base (constraint)
    - collocate_word_num: the number of words in the collocate (constraint)
    - max_data_limit: the max data limit
    - max_instance_num_per_category: the max instance number per category
    - mask_collocate: whether to mask the collocate
    - verbose: whether to print the statistics

    Returns:
    - output_data: the prepared data
    """
    df = pd.read_csv(data_path, sep="\t")

    category2freq = dict()
    instances_processed = []
    for i in range(len(df)):
        base = df.iloc[i, 1].replace("_", "")
        collocate_idx = int(df.iloc[i, 6]) - 1
        collocate = df.iloc[i, 4].replace("_", "")
        if base_word_num and len(base.split()) > base_word_num:
            continue
        if collocate_word_num and len(collocate.split()) > collocate_word_num:
            continue
        collocation = (
            f"{base} {collocate}"
            if int(df.iloc[i, 6]) > int(df.iloc[i, 7])
            else f"{collocate} {base}"
        )
        label = df.iloc[i, 5]
        if category2freq.get(label, 0) >= max_instance_num_per_category:
            continue
        if mask_collocate:
            words = df.iloc[i, 10].split()
            if words[collocate_idx] != collocate:
                print(
                    f"id: {str(i).zfill(3)}\twords[collocate_idx] ({words[collocate_idx]}) != collocate ({collocate})."
                    "\tplease compile this instance manually."
                )
                # continue
            else:
                words[collocate_idx] = "[MASK]"
            context = " ".join(words)
        else:
            context = df.iloc[i, 10]
        instance = {
            "id": str(i).zfill(3),
            "base": base,
            "collocate": collocate,
            "collocation": collocation,
            "label": df.iloc[i, 5],
            "context": context,
        }
        instances_processed.append(instance)
        category2freq[label] = category2freq.get(label, 0) + 1

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        # dump `instances_processed`
        with open(dump_data_path, "w") as f:
            for instance in instances_processed:
                f.write(
                    f"{instance['id']}\t{instance['base']}\t{instance['collocate']}\t{instance['collocation']}\t{instance['label']}\t{instance['context']}\n"
                )


def prepare_data_collocation_categorization(
    data_path: str,
    dump_data_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Prepare the collocation categorization data for evaluation.

    Args:
    - data_path: the path of the data file
    - dump_data_path: the path of the prepared data file

    Returns:
    - output_data: the prepared data
    """
    label_map = {
        "Magn": 0,
        "AntiMagn": 1,
        "Ver": 2,
        "AntiVer": 3,
        "Bon": 4,
        "AntiBon": 5,
        "Son": 6,
        "Oper1": 7,
    }
    df = pd.read_csv(data_path, sep="\t")
    output_data = []
    for i in range(len(df)):
        output_data.append(
            {
                "id": str(i).zfill(3),
                "base": df.iloc[i, 1],
                "collocate": df.iloc[i, 2],
                "collocation": df.iloc[i, 3],
                "category": df.iloc[i, 4],
                "label": label_map[df.iloc[i, 4]],
                "context": df.iloc[i, 5].replace("[MASK]", df.iloc[i, 2]),
            }
        )

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        # dump `output_data`
        with open(dump_data_path, "w") as f:
            for item in output_data:
                f.write(
                    f"{item['id']}\t{item['base']}\t{item['collocate']}\t{item['collocation']}\t{item['category']}\t{item['label']}\t{item['context']}\n"
                )

    return output_data


def prepare_noun_compound_interpretation(
    data_path: str,
    dump_data_path: Optional[str] = None,
    max_data_limit: int = 110,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Prepare the noun compound interpretation data for evaluation.

    Args:
    - data_path: the path of the data file
    - dump_data_path: the path of the prepared data file
    - max_data_limit: the max data limit
    - verbose: whether to print the statistics

    Returns:
    - output_data: the prepared data
    """
    df = pd.read_csv(data_path, sep=",")
    output_data = []
    for i in range(len(df)):
        noun_compound = df.iloc[i, 1]
        paraphrases = ast.literal_eval(df.iloc[i, 2])
        output_data.append(
            {
                "id": str(i).zfill(3),
                "noun_compound": noun_compound,
                "paraphrases": paraphrases,
            }
        )

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        # dump `output_data`
        with open(dump_data_path, "w") as f:
            for item in output_data:
                f.write(
                    f"{item['id']}\t{item['noun_compound']}\t{item['paraphrases']}\n"
                )

    if verbose:
        print("Total instances:", len(output_data))

    return output_data


def prepare_noun_compound_extraction(
    data_path: str,
    dump_data_path: Optional[str] = None,
    max_data_limit: int = 110,
    min_context_size: int = 8,
    max_context_size: int = 32,
    dedup: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Prepare the noun compound extraction data for evaluation.

    Args:
    - data_path: the path of the data file
    - dump_data_path: the path of the prepared data file
    - max_data_limit: the max data limit
    - min_context_size: the min context size
    - max_context_size: the max context size
    - dedup: whether to deduplicate the data by the whole collocation
    - verbose: whether to print the statistics

    Returns:
    - output_data: the prepared data
    """
    output_data = []
    with open(data_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        for line in lines:
            instance = json.loads(line)
            if (
                not instance["is_compositional"]
                or instance["comment"] != "{}"
                or not instance["is_context_needed"]
                or len(output_data) >= max_data_limit
            ):
                continue
            context = instance["sentence"]
            word_list = context.split()
            nnp_idx = instance["nnp_index"]
            nn_idx = instance["nn_index"]
            nnp = instance["nnp"]
            nn = instance["nn"]
            if (
                nnp_idx + 1 > len(word_list)
                or nn_idx + 1 > len(word_list)
                or word_list[nnp_idx] != nnp
                or word_list[nn_idx] != nn
                or len(word_list) < min_context_size
                or len(word_list) > max_context_size
            ):
                continue
            nc = f"{nnp} {nn}"
            output_data.append(
                {
                    "context": context,
                    "nnp_index": nnp_idx,
                    "nn_index": nn_idx,
                    "nc": nc,
                    "paraphrase": instance["explicit_relation"],
                }
            )

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        # dump `output_data`
        with open(dump_data_path, "w") as f:
            for item in output_data:
                f.write(
                    f"{item['context']}\t{item['nnp_index']}\t{item['nn_index']}\t{item['nc']}\t{item['paraphrase']}\n"
                )

    if verbose:
        print("Total instances:", len(output_data))

    return output_data


def prepare_noun_compound_compositionality(
    nc_data_path: str,
    sent_data_path: str,
    dump_data_path: str,
    max_data_limit: int,
) -> List[Dict[str, Any]]:
    """
    Prepare the noun compound compositionality data for evaluation.

    Args:
    - nc_data_path: the path of the noun compound data file
    - sent_data_path: the path of the sentence data file
    - dump_data_path: the path of the prepared data file
    - max_data_limit: the max data limit

    Returns:
    - output_data: the prepared data
    """
    nc_data = pd.read_csv(nc_data_path, sep="\t")
    sent_data = pd.read_csv(sent_data_path, sep=",")
    abbr2name = {
        "C": "Compositional",
        "PC": "Partly compositional",
        "NC": "Non-compositional",
        "None": "None of the above",
    }
    abbr2idx = {
        "C": 0,
        "PC": 1,
        "NC": 2,
        "None": 3,
    }
    label_map = {
        "Compositional": 0,
        "Partly compositional": 1,
        "Non-compositional": 2,
        "None of the above": 3,
    }
    abbr2option = {
        "C": "A",
        "PC": "B",
        "NC": "C",
        "None": "D",
    }
    label2option = {
        "Compositional": "A",
        "Partly compositional": "B",
        "Non-compositional": "C",
        "None of the above": "D",
    }
    option2label = {v: k for k, v in label2option.items()}
    options_str = "Compositional[SEP]Partly compositional[SEP]Non-compositional[SEP]None of the above"

    # load `nc_data` and `sent_data`
    nc_set = set(nc_data["compound"].tolist())
    sent_nc_set = set(sent_data["compound"].tolist())

    # find intersetion of `nc_set` and `sent_nc_set`
    intersection_nc_set = nc_set.intersection(sent_nc_set)
    # print(len(intersection_nc_set))

    idx = 0
    output_data = []
    output_data_ft = []
    for i in range(len(nc_data)):
        # shuffle `abbr2option`
        abbr2option = dict(
            zip(
                list(random.sample(list(abbr2option.keys()), len(abbr2option))),
                abbr2option.values(),
            )
        )
        option2abbr = {v: k for k, v in abbr2option.items()}
        noun_compound = nc_data.iloc[i, 0]
        noun_compound_type = nc_data.iloc[i, 1]
        for j in range(len(sent_data)):
            if sent_data.iloc[j, 0] == noun_compound:
                sentence_1 = sent_data.iloc[j, 1]
                sentence_2 = sent_data.iloc[j, 2]
                sentence_3 = sent_data.iloc[j, 3]
                sentence = (
                    sentence_1
                    if "http" not in sentence_1
                    else sentence_2
                    if "http" not in sentence_2
                    else sentence_3
                    if "http" not in sentence_3
                    else ""
                )
                if sentence == "":
                    print(f"detect empty sentence: {noun_compound}")
                    break
                output_data.append(
                    {
                        "id": str(idx).zfill(3),
                        "noun_compound": noun_compound,
                        "sentence": sentence,
                        "A": abbr2name[option2abbr["A"]],
                        "B": abbr2name[option2abbr["B"]],
                        "C": abbr2name[option2abbr["C"]],
                        "D": abbr2name[option2abbr["D"]],
                        "target": abbr2option[noun_compound_type],
                    }
                )
                idx += 1
                output_data_ft.append(
                    f"{sentence}[SEP]{noun_compound}\t{options_str}\t{abbr2idx[noun_compound_type]}\n"
                )

    output_data_ft_train, output_data_ft_valid_test = train_test_split(
        output_data_ft, test_size=0.5, random_state=42
    )
    output_data_ft_valid, output_data_ft_test = train_test_split(
        output_data_ft_valid_test, test_size=0.8, random_state=42
    )

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        # dump `output_data`
        with open(dump_data_path, "w") as f:
            for item in output_data:
                f.write(
                    f"{item['id']}\t{item['noun_compound']}\t{item['sentence']}\t{item['A']}\t{item['B']}\t{item['C']}\t{item['D']}\t{item['target']}\n"
                )
        # dump `output_data_ft`
        with open(dump_data_path.replace(".tsv", "_ft_train.tsv"), "w") as f:
            f.writelines(output_data_ft_train)
        with open(dump_data_path.replace(".tsv", "_ft_valid.tsv"), "w") as f:
            f.writelines(output_data_ft_valid)
        with open(dump_data_path.replace(".tsv", "_ft_test.tsv"), "w") as f:
            f.writelines(output_data_ft_test)
        # dump `label_map`
        with open(
            dump_data_path.rsplit("/", maxsplit=1)[0] + "/labelmap.tsv", "w"
        ) as f:
            for label, idx in label_map.items():
                f.write(f"{label}\t{idx}\n")

    return output_data


def prepare_vmwe_identification(
    data_path: str,
    dump_data_path: str,
    max_data_limit: int,
    include_one_vmwe: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Prepare the VMWE identification data for evaluation.

    Args:
    - data_path: the path of the data file
    - dump_data_path: the path of the prepared data file
    - max_data_limit: the max data limit
    - include_one_vmwe: whether to include the instance with only one VMWE
    - verbose: whether to print the statistics

    Returns:
    - output_data: the prepared data
    """

    def match_text(s: str, pattern: str) -> List[str]:
        return re.search(pattern, s).group(0).replace("# text = ", "").strip()

    def parse_vmwe_part(s: str) -> Tuple[str, str]:
        return [e.strip() for e in s.strip().split(":")]

    output_data = []
    category2freq = dict()

    chunks = [
        chunk
        for chunk in open(data_path, "r").read().split("\n\n")
        if chunk.strip() != ""
    ]

    idx = 0
    for chunk in chunks:
        label2words: Dict[str, List] = dict()
        vmwe_orders = []
        vmwe_words = []
        vmwe_categories = []
        order2label = dict()
        text = match_text(chunk, r"# text = .*\n")
        lines = [
            line
            for line in chunk.split("\n")
            if line.strip() != "" and "# " not in line
        ]
        for line in lines:
            (
                id_,
                word,
                lemma,
                upos,
                xpos,
                feats,
                head,
                deprel,
                deps,
                misc,
                parseme_mwe,
            ) = line.split("\t")
            # skip lines without part of mwe
            if parseme_mwe == "*":
                continue
            # process lines with part of mwe
            if ":" in parseme_mwe and ";" not in parseme_mwe:
                ## examples: [3:VPC.full]
                order, category = parse_vmwe_part(parseme_mwe)
                vmwe_words.append(word)
                vmwe_orders.append(order)
                vmwe_categories.append(f"{order}-{category}")
                order2label[order] = f"{order}-{category}"
            elif ":" in parseme_mwe and ";" in parseme_mwe:
                ## examples: [2;3:VPC.full, 2:LVC;3:VPC.full]
                for part in parseme_mwe.split(";"):
                    if ":" in part:
                        order, category = parse_vmwe_part(part)
                        vmwe_words.append(word)
                        vmwe_orders.append(order)
                        vmwe_categories.append(f"{order}-{category}")
                        order2label[order] = f"{order}-{category}"
                    else:
                        order = part
                        vmwe_words.append(word)
                        vmwe_orders.append(order)
                        label = order2label[order]
                        vmwe_categories.append(label)
            elif ":" not in parseme_mwe and ";" in parseme_mwe:
                ## examples: [2;3]
                vmwe_words.append(word)
                vmwe_words.append(word)
                orders = parseme_mwe.split(";")
                for order in orders:
                    label = order2label[order]
                    vmwe_categories.append(label)
                    vmwe_orders.append(order)
            elif ":" not in parseme_mwe and ";" not in parseme_mwe:
                ## examples: [2]
                order = parseme_mwe
                label = order2label[order]
                vmwe_words.append(word)
                vmwe_orders.append(order)
                vmwe_categories.append(label)

        for label, word in zip(vmwe_categories, vmwe_words):
            if label not in label2words:
                label2words[label] = []
            label2words[label].append(word)

        if label2words:
            vmwes = [" ".join(words) for words in label2words.values()]
            labels = [
                label.split("-", maxsplit=1)[-1].strip() for label in label2words.keys()
            ]
            if include_one_vmwe:
                if "VID" in labels:
                    idx_label = labels.index("VID")
                elif "LVC.full" in labels:
                    idx_label = labels.index("LVC.full")
                elif "LVC.cause" in labels:
                    idx_label = labels.index("LVC.cause")
                elif "VPC.full" in labels:
                    idx_label = labels.index("VPC.full")
                elif "VPC.semi" in labels:
                    idx_label = labels.index("VPC.semi")
                else:
                    # drop the instances of which only include MVC and IAV
                    continue
                vmwes = [vmwes[idx_label]]
                labels = [labels[idx_label]]
            output_data.append(
                {
                    "id": str(idx).zfill(3),
                    "text": text,
                    "vmwes": vmwes,
                    "label": labels,
                }
            )
            for label in labels:
                category2freq[label] = category2freq.get(label, 0) + 1
            # detect whether each word of vmwe is in text or not
            for vmwe in vmwes:
                for word in vmwe.split():
                    assert word in text
            idx += 1

    if verbose:
        pprint(output_data, indent=4)
        pprint(category2freq, indent=4)

    if dump_data_path:
        if not os.path.exists(os.path.dirname(dump_data_path)):
            os.makedirs(os.path.dirname(dump_data_path))
        with open(dump_data_path, "w") as f:
            for item in output_data:
                f.write(
                    f"{item['id']}\t{item['text']}\t{','.join(item['vmwes'])}\t{','.join(item['label'])}\n"
                )


def prepare_data_collocation_categorization_scaling(
    train_data_path: str,
    valid_data_path: str,
    test_data_path: str,
    example_data_path: str,
    label_path: str,
    taxonomy_path: str,
    dump_data_dir: Optional[str] = None,
    max_data_limit: int = 30,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Prepare the collocation categorization data for evaluation.

    Args:
    - train_data_path: the path of the train data file
    - valid_data_path: the path of the valid data file
    - test_data_path: the path of the test data file
    - example_data_path: the path of the example data file
    - label_path: the path of the label file
    - taxonomy_path: the path of the taxonomy file
    - dump_data_dir: the directory of the prepared data file
    - max_data_limit: the max data limit
    - verbose: whether to print the statistics

    Returns:
    - output_data: the prepared data
    """

    def construct_label_map(label_path: str) -> Dict[str, int]:
        label2id = dict()
        id2label = dict()
        with open(label_path, "r") as f:
            lines = [l.strip() for l in f.readlines()]
            for line in lines:
                label, id_ = line.split("\t")
                label2id[label] = id_
                id2label[id_] = label

        return label2id, id2label

    def construct_adaptive_taxonomy(
        taxonomy_path: str, categories: List[str], shot_num: int
    ) -> str:
        taxonomy_str = ""
        with open(taxonomy_path, "r") as f:
            lines = [l.strip() for l in f.readlines()]
            for idx, line in enumerate(lines):
                if idx == 0:
                    taxonomy_str += (
                        line.replace("Examples\t", "") + "\n"
                        if shot_num == 0
                        else line + "\n"
                    )
                    continue
                label, examples_str, gloss = line.split("\t")
                if shot_num > 0 and idx != 0:
                    examples_str = ",".join(
                        [e.strip() for e in examples_str.split(",")][:shot_num]
                    )
                if label in categories:
                    taxonomy_str += (
                        f"{label}\t{examples_str}\t{gloss}\n"
                        if shot_num > 0
                        else f"{label}\t{gloss}\n"
                    )

        return taxonomy_str

    instances_train = []
    instances_valid = []
    instances_test = []
    shot_nums = [0, 3, 5]
    category_nums = [1, 2, 4, 8, 16]
    label2id, id2label = construct_label_map(label_path)
    labels = list(label2id.keys())  # 16 labels

    with open(train_data_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        for line in lines:
            instance = line.split("\t")
            instances_train.append(
                {
                    "context": instance[0],
                    "collocation": instance[1],
                    "id": instance[2],
                    "label": id2label[instance[2]],
                }
            )

    with open(valid_data_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        for line in lines:
            instance = line.split("\t")
            instances_valid.append(
                {
                    "context": instance[0],
                    "collocation": instance[1],
                    "id": instance[2],
                    "label": id2label[instance[2]],
                }
            )

    with open(test_data_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        for line in lines:
            instance = line.split("\t")
            instances_test.append(
                {
                    "context": instance[0],
                    "collocation": instance[1],
                    "id": instance[2],
                    "label": id2label[instance[2]],
                }
            )

    for category_num in category_nums:
        # run three times with three fixed different seeds
        for seed_id, seed in enumerate([21, 42, 84]):
            random.seed(seed)

            categories = sorted(random.sample(labels, k=category_num))

            # construct train and valid data with categories
            instances_train_categories = [
                i for i in instances_train if i["label"] in categories
            ]
            instances_valid_categories = [
                i for i in instances_valid if i["label"] in categories
            ]

            # random select k instantces for each category
            instances_test_sampled = []
            for category in categories:
                instances_test_categories = [
                    i
                    for i in instances_test
                    if i["label"] == category and len(i["context"].split()) < 64
                ]
                instances_test_categories = random.sample(
                    instances_test_categories, max_data_limit
                )
                instances_test_sampled += instances_test_categories

            # dump data
            if dump_data_dir:
                sub_data_dir = f"{dump_data_dir}/{str(category_num)}-{str(seed_id)}"
                os.makedirs(sub_data_dir, exist_ok=True)
                shutil.copy(example_data_path, f"{sub_data_dir}/example.tsv")
                shutil.copy(label_path, f"{sub_data_dir}/labelmap.tsv")
                for shot_num in shot_nums:
                    taxonomy_content = construct_adaptive_taxonomy(
                        taxonomy_path, categories, shot_num
                    )
                    with open(
                        f"{sub_data_dir}/taxonomy_{str(shot_num)}-shot.tsv", "w"
                    ) as f:
                        f.write(taxonomy_content)
                with open(f"{sub_data_dir}/test.tsv", "w") as f:
                    for instance in instances_test_sampled:
                        f.write(
                            f"{instance['id']}\tNone\tNone\t{instance['collocation']}\t{instance['label']}\tNone\t{instance['context']}\n"
                        )
                with open(f"{sub_data_dir}/train_ft.tsv", "w") as f:
                    for instance in instances_train_categories:
                        f.write(
                            f"{instance['context']}\t{instance['collocation']}\t{instance['id']}\n"
                        )
                with open(f"{sub_data_dir}/valid_ft.tsv", "w") as f:
                    for instance in instances_valid_categories:
                        f.write(
                            f"{instance['context']}\t{instance['collocation']}\t{instance['id']}\n"
                        )
                with open(f"{sub_data_dir}/test_ft.tsv", "w") as f:
                    for instance in instances_test_sampled:
                        f.write(
                            f"{instance['context']}\t{instance['collocation']}\t{instance['id']}\n"
                        )


def extract_collocation_list(data_path: str, dump_data_path: str):
    with open(data_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        collocation_list = []
        for line in lines:
            collocation_list.append(line.split("\t")[3])

    with open(dump_data_path, "w") as f:
        for collocation in collocation_list:
            f.write(f"{collocation}\n")


if __name__ == "__main__":
    # [Task 1] Idiomatic Expression Detection (IED)
    # idiom_detection_examples = prepare_data_idiom_detection(
    # refer_data_path="dataset/idiom_detection/reference_data.csv",
    # idiom_data_path="dataset/idiom_detection/idiom_data.csv",
    # )

    # [Task 2] Idiomatic Expression Extraction (IEE)
    # idiom_extraction_examples = prepare_data_idiom_extraction(
    # data_path="dataset/idiom_extraction/dev_english.tsv",
    # dump_data_path="dataset/idiom_extraction/prepared/idiom_extraction_prepared.tsv",
    # )

    # [Task 3-1] Idiomatic Expression Paraphrase (IEP)
    # idiom_paraphrase_examples = prepare_data_idiom_paraphrase_1(
    # idiom_data_path="dataset/idiom_paraphrase/data_cleaned.csv"
    # )

    # [Task 3-2] Idiomatic Expression Paraphrase (IEP)
    # idiom_paraphrase_examples = prepare_data_idiom_paraphrase_2(
    # idiom_data_path="LexBench/resources/FigurativeNarrativeBenchmark/data/idiom/all.jsonl",
    # idiom_data_ref_path="dataset/idiom_paraphrase/prepared/idiom_paraphrase_prepared.tsv",
    # dump_data_path="dataset/idiom_paraphrase/prepared/idiom_paraphrase_prepared_2.tsv",
    # )

    # [Task 4] Lexical Collocate Retrieval (LCR)
    # sample_data_collocate_retrieval(
    # data_path="dataset/collocate_retrieval/Collocations_en.tsv",
    # dump_data_path="dataset/collocate_retrieval/Collocations_en_test.tsv",
    # category_num=8,
    # max_instance_num_per_category=40,
    # )
    # collocate_retrieval_examples = prepare_data_collocate_retrieval(
    # data_path="dataset/collocate_retrieval/Collocations_en_test.tsv",
    # dump_data_path="dataset/collocate_retrieval/prepared/collocate_retrieval_prepared_new.tsv",
    # base_word_num=3,
    # collocate_word_num=10,
    # )

    # [Task 5] Lexical Collocate Categorization (LCC)
    # collocation_categorization_examples = prepare_data_collocation_categorization(
    # data_path="dataset/collocate_retrieval/prepared/collocate_retrieval_prepared.tsv",
    # dump_data_path="dataset/collocation_categorization/prepared/collocate_categorization_prepared.tsv",
    # )

    # [Task 6] Lexical Collocate Extraction (LCE)
    # collocate_extraction_examples = prepare_data_collocation_extraction(
    # data_path="dataset/collocation_extraction/Collocate_en.tsv",  # FIXME
    # dump_data_path="dataset/collocate_extraction/prepared/collocate_extraction_prepared.tsv",
    # max_data_limit=1000,
    # )

    # [Task 7] Noun Compound Compositionality (NCC)
    noun_compound_relation_examples = prepare_noun_compound_compositionality(
        nc_data_path="dataset/noun_compound_compositionality/data_en.tsv",
        sent_data_path="dataset/noun_compound_compositionality/sentids_en.csv",
        dump_data_path="dataset/noun_compound_compositionality/prepared/noun_compound_compositionality_prepared.tsv",
        max_data_limit=1000,
    )

    # [Task 8] Noun Compound Extraction (NCE)
    # noun_compound_extraction_examples = prepare_noun_compound_extraction(
    # data_path="dataset/noun_compound_extraction/test.jsonl",
    # dump_data_path="dataset/noun_compound_extraction/prepared/noun_compound_extraction_prepared.tsv",
    # max_data_limit=1000,
    # )

    # [Task 9] Noun Compound Interpretation (NCI)
    # noun_compound_interpretation_examples = prepare_noun_compound_interpretation(
    # data_path="dataset/noun_compound_interpretation/test_df.csv",
    # dump_data_path="dataset/noun_compound_interpretation/prepared/noun_compound_interpretation_prepared.tsv",
    # )
    # noun_compound_interpretation_examples = prepare_noun_compound_interpretation(
    # data_path="dataset/noun_compound_interpretation/valid_df.csv",
    # dump_data_path="dataset/noun_compound_interpretation/prepared/examples.tsv",
    # )

    # [Task 10] VMWE Identification (VPC/LVC/VID)
    # vmwe_identification_examples = prepare_vmwe_identification(
    # data_path="dataset/verbal_mwe_extraction/EN/test-sample.cupt",
    # data_path="dataset/verbal_mwe_extraction/EN/train.cupt",
    # data_path="dataset/verbal_mwe_extraction/EN/dev.cupt",
    # data_path="dataset/verbal_mwe_extraction/EN/test.cupt",
    # dump_data_path="dataset/verbal_mwe_extraction/prepared/vmwe_identification_train_unique_prepared.tsv",
    # max_data_limit=10000,
    # include_one_vmwe=True,
    # verbose=True,
    # )

    # [Scaling Exp] Collocation Categorization
    # extract_collocation_list(
    # data_path="dataset/collocation_paraphrase/collocation_paraphrase.tsv",
    # dump_data_path="dataset/collocation_paraphrase/collocations.tsv",
    # )

    # collocation_categorization_examples_scaling = prepare_data_collocation_categorization_scaling(
    # train_data_path="LexBench/resources/lexicalcollocations/data/lf_clf/train_data.tsv",
    # valid_data_path="LexBench/resources/lexicalcollocations/data/lf_clf/validation_data.tsv",
    # test_data_path="LexBench/resources/lexicalcollocations/data/lf_clf/test_data.tsv",
    # example_data_path="LexBench/scripts/dataset/collocation_categorization/prepared/examples.tsv",
    # label_path="LexBench/resources/lexicalcollocations/data/lf_clf/labelmap.tsv",
    # taxonomy_path="LexBench/exp/taxonomy/SEM_REL_CATEGORY_16.txt",
    # dump_data_dir="dataset/collocation_categorization",
    # )
