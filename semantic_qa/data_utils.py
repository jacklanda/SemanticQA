# -*- coding: utf-8 -*-

import re
import ast
import random
from argparse import Namespace
from typing import List, Dict, Any, Optional

from rich import print_json
from rich.console import Console
from type import LF_CATEGORY_ID2LABEL_MAP_8, LF_CATEGORY_LIST_8


random.seed(42)
console = Console()

def get_noun_compound_data(
    data_path: str,
    task_type: str,
    max_num_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    examples = []
    if task_type == "noun-compound-compositionality":
        for idx, line in enumerate(open(data_path, "r", encoding="utf-8")):
            if idx == 0:
                continue
            if len(examples) >= max_num_limit:
                break
            line = line.strip()
            if line == "":
                continue
            if len(line.split("\t")) != 8:
                continue
            id_, noun_compound, context, A, B, C, D, target = [
                item.strip() for item in line.split("\t")
            ]
            examples.append(
                {
                    "id": id_,
                    "noun_compound": noun_compound,
                    "context": f"{context}\nOptions: A: {A}\tB: {B}\tC: {C}\tD: {D}",
                    "A": A,
                    "B": B,
                    "C": C,
                    "D": D,
                    "target": target,
                }
            )
    elif task_type == "noun-compound-interpretation":
        for line in open(data_path, "r", encoding="utf-8"):
            if len(examples) >= max_num_limit:
                break
            line = line.strip()
            if line == "":
                continue
            line = line.split("\t")
            if len(line) != 3:
                continue
            noun_compound = line[1]
            references = ast.literal_eval(line[2])
            examples.append(
                {
                    "noun_compound": noun_compound,
                    "references": references,
                }
            )
    elif task_type == "noun-compound-extraction":
        for line in open(data_path, "r", encoding="utf-8"):
            if len(examples) >= max_num_limit:
                break
            line = line.strip()
            if line == "":
                continue
            line = line.split("\t")
            if len(line) != 5:
                continue
            context = line[0]
            noun_compound = line[3]
            examples.append(
                {
                    "context": context,
                    "noun_compound": noun_compound,
                }
            )

    random.shuffle(examples)

    return examples


def get_collocation_data(
    data_path: str,
    task_type: str,
    max_num_limit: int,
) -> List[Dict[str, Any]]:
    """
    Get data example from data path
    :param data_path: data path
    :param max_num_limit: max number of data
    :return: data example
    """
    examples = []
    if task_type == "collocate-retrieval":
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(examples) >= max_num_limit:
                    break
                line = line.strip()
                if line == "":
                    continue
                line = line.split("\t")
                if len(line) != 6:
                    continue
                base = line[1].replace("_", "")
                collocate = line[2].replace("_", "")
                collocation = line[3]
                label = line[4]
                context = line[5]
                examples.append(
                    {
                        "base": base,
                        "collocate": collocate,
                        "collocation": collocation,
                        "label": label,
                        "context": context,
                    }
                )
    elif task_type == "collocation-categorization":
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(examples) >= max_num_limit:
                    break
                line = line.strip()
                if line == "":
                    continue
                line = line.split("\t")
                if len(line) != 7:
                    continue
                base = line[1]
                collocate = line[2]
                collocation = line[3]
                label = line[4]
                label_id = line[5]
                context = line[6]
                examples.append(
                    {
                        "base": base,
                        "collocate": collocate,
                        "collocation": collocation,
                        "label": label,
                        "label_id": label_id,
                        "context": context,
                    }
                )
    elif task_type == "collocation-extraction":
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(examples) >= max_num_limit:
                    break
                line = line.strip()
                if line == "":
                    continue
                line = line.split("\t")
                if len(line) != 7:
                    continue
                keyword = line[1].replace("_", "")
                value = line[2].replace("_", "")
                collocation = line[3]
                label = line[4]
                context = line[6]
                examples.append(
                    {
                        "keyword": keyword,
                        "value": value,
                        "collocation": collocation,
                        "label": label,
                        "context": context,
                    }
                )
    elif task_type == "collocation-interpretation":
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(examples) >= max_num_limit:
                    break
                line = line.strip()
                if line == "":
                    continue
                line = line.split("\t")
                if len(line) != 8:  
                    continue
                base = line[1]
                collocate = line[2]
                collocation = line[3]
                label = line[4]
                label_id = line[5]
                context = line[6]
                paraphrases = line[7].split("#####")

                examples.append({
                    "base": base,
                    "collocate": collocate,
                    "collocation": collocation,
                    "label": label,
                    "label_id": label_id,
                    "context": context,
                    "paraphrases": paraphrases
                })

    elif task_type == "collocation-scaling":
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(examples) >= max_num_limit:
                    break
                line = line.strip()
                if line == "":
                    continue
                line = line.split("\t")
                if len(line) != 7:  
                    continue
                label_id = line[0]
                collocation = line[3]
                label = line[4]
                context = line[6]
                examples.append({
                    "collocation": collocation,
                    "label":label,
                    "label_id": label_id,
                    "context": context
                })

    # random.shuffle(examples)

    return examples


def get_idiom_data(
    data_path: str, task_type: str, max_num_limit: int
) -> List[Dict[str, Any]]:
    """
    Load idiom data example from data path

    Args:
    - data_path: data path
    - task_type: task type
    - max_num_limit: max number of data

    Returns:
    - data examples
    """
    data_examples = []

    if task_type == "idiom-detection":
        with open(data_path, "r", encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()][1:]
            for line in lines:
                id_, context, idiom, A, B, C, D, target = [
                    item.strip() for item in line.split("\t")
                ]
                data_examples.append(
                    {
                        "id": id_,
                        "context": f"{context}\nA: {A}\tB: {B}\tC: {C}\tD: {D}",
                        "idiom": idiom,
                        "A": A,
                        "B": B,
                        "C": C,
                        "D": D,
                        "target": target,
                    }
                )
    elif task_type == "idiom-extraction":
        with open(data_path, "r", encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()][1:]
            for line in lines:
                context, idiom = [item.strip() for item in line.split("\t")]
                data_examples.append({"context": context, "idiom": idiom})
    elif task_type == "idiom-paraphrase":
        with open(data_path, "r", encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()][1:]
            for line in lines:
                id_, idiom, paraphrase, context_idiomatic, context_literal = [
                    item.strip() for item in line.split("\t")
                ]
                data_examples.append(
                    {
                        "id": id_,
                        "idiom": idiom,
                        "paraphrase": paraphrase,
                        "context_idiomatic": context_idiomatic,
                        "context_literal": context_literal,
                    }
                )
    else:
        raise NotImplementedError(
            "Only support following list of tasks for now: [idiom-detection, idiom-extraction, idiom-paraphrase]."
        )

    random.shuffle(data_examples)

    return data_examples


def get_vmwe_data(
    data_path: str, task_type: str, max_num_limit: int
) -> List[Dict[str, Any]]:
    """Load VMWE data example from data path

    Args:
    - data_path: data path
    - task_type: task type
    - max_num_limit: max number of data

    Returns:
    - data examples
    """
    data_examples = []
    # https://parsemefr.lis-lab.fr/parseme-st-guidelines/1.3/?page=010_Definitions_and_scope
    # TODO: move these definitions to data module.
    vpc_def = "Verb-particle construction (VPC), sometimes called phrasal verb or phrasal-prepositional verb. The meaning of the VPC is fully or partly non-compositional."
    lvc_def = "Light verb constructions (LVC) are formed by a verb v and a (single or compound) noun n, which either directly depends on v (and possibly contains a case marker or a postposition), or is introduced by a preposition."
    vid_def = "Verbal idioms constitute a universal category. A verbal idiom (VID) has at least two lexicalized components including a head verb and at least one of its dependents. The dependent can be of different types."
    if task_type == "vmwe-extraction":
        with open(data_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
            for line in lines:
                # console.print("line:", line)
                # exit()
                id_, context, vmwe, label = [item.strip() for item in line.split("\t")]
                if "vpc" in label.lower():
                    oracle = vpc_def
                elif "lvc" in label.lower():
                    oracle = lvc_def
                elif "vid" in label.lower():
                    oracle = vid_def
                data_examples.append(
                    {
                        "id": id_,
                        "context": context,
                        "vmwe": vmwe,
                        "label": label,
                        "oracle": oracle,
                    }
                )
    else:
        raise NotImplementedError(
            "Only support following list of tasks for now: [vmwe-extraction]."
        )

    random.shuffle(data_examples)

    return data_examples


def get_sequential_task_data(
    data_path: str, task_type: str, max_num_limit: int
) -> List[Dict[str, Any]]:
    """Load data from data path for multi-step reasoning

    Args:
    - data_path: data path
    - task_type: task type
    - max_num_limit: max number of data

    Returns:
    - data
    """
    data_examples = []
    if task_type == "sequential-task-idiom-extraction-and-judgment":
       with open(data_path, "r", encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()][1:]
            for line in lines:
                id_, context, idiom, A, B, C, D, target = [
                    item.strip() for item in line.split("\t")
                ]
                data_examples.append(
                    {
                        "id": id_,
                        "context": f"{context}\nA: {A}\tB: {B}\tC: {C}\tD: {D}",
                        "idiom": idiom,
                        "A": A,
                        "B": B,
                        "C": C,
                        "D": D,
                        "target": target,
                    }
                )

    elif task_type == "sequential-task-collocate-extraction-and-judgment":
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(data_examples) >= max_num_limit:
                    break
                line = line.strip()
                if line == "":
                    continue
                line = line.split("\t")
                if len(line) != 7:  
                    continue
                label_id = line[0]
                collocation = line[3]
                label = line[4]
                context = line[6]
                data_examples.append({
                    "collocation": collocation,
                    "label":label,
                    "label_id": label_id,
                    "context": context
                })

    elif task_type == "sequential-task-noun-compound-extraction-and-judgment":
        for idx, line in enumerate(open(data_path, "r", encoding="utf-8")):
            if idx == 0:
                continue
            if len(data_examples) >= max_num_limit:
                break
            line = line.strip()
            if line == "":
                continue
            if len(line.split("\t")) != 8:
                continue
            id_, noun_compound, context, A, B, C, D, target = [
                item.strip() for item in line.split("\t")
            ]
            data_examples.append(
                {
                    "id": id_,
                    "noun_compound": noun_compound,
                    "context": f"{context}\nOptions: A: {A}\tB: {B}\tC: {C}\tD: {D}",
                    "A": A,
                    "B": B,
                    "C": C,
                    "D": D,
                    "target": target,
                }
            )

    elif task_type == "sequential-task-idiom-extraction-and-interpretation":
        with open(data_path, "r", encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()][1:]
            for line in lines:
                id_, idiom, paraphrase, context_idiomatic, context_literal = [
                    item.strip() for item in line.split("\t")
                ]
                data_examples.append(
                    {
                        "id": id_,
                        "idiom": idiom,
                        "paraphrase": paraphrase,
                        "context_idiomatic": context_idiomatic,
                        "context_literal": context_literal,
                    }
                )
    
    elif task_type == "sequential-task-collocate-extraction-and-interpretation":
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(data_examples) >= max_num_limit:
                    break
                line = line.strip()
                if line == "":
                    continue
                line = line.split("\t")
                if len(line) != 8:  
                    continue
                base = line[1]
                collocate = line[2]
                collocation = line[3]
                label = line[4]
                label_id = line[5]
                context = line[6]
                paraphrases = line[7].split("#####")

                data_examples.append({
                    "base": base,
                    "collocate": collocate,
                    "collocation": collocation,
                    "label": label,
                    "label_id": label_id,
                    "context": context,
                    "paraphrases": paraphrases
                })
    
    elif task_type == "sequential-task-noun-compound-extraction-and-interpretation":
        for line in open(data_path, "r", encoding="utf-8"):
            if len(data_examples) >= max_num_limit:
                break
            line = line.strip()
            if line == "":
                continue
            line = line.split("\t")
            if len(line) != 3:
                continue
            noun_compound = line[1]
            references = ast.literal_eval(line[2])
            data_examples.append(
                {
                    "noun_compound": noun_compound,
                    "references": references,
                }
            )

    random.shuffle(data_examples)
    return data_examples



def load_data_example(
    data_path: str, task_type: str, max_num_limit: int
) -> List[Dict[str, Any]]:
    """
    Load data example from data path

    Args:
    - data_path: data path
    - task_type: task type
    - max_num_limit: max number of data

    Returns:
    - data examples
    """
    data_examples = []

    if task_type in [
        "idiom-detection", 
        "sequential-task-idiom-extraction-and-judgment"       
    ]:
        with open(data_path, "r") as f:
            lines = [line.strip() for line in f.readlines()][1:]
            for line in lines:
                id_, context, idiom, A, B, C, D, target = [
                    item.strip() for item in line.split("\t")
                ]
                data_examples.append(
                    {
                        "id": id_,
                        "context": f"{context}\nA: {A}\tB: {B}\tC: {C}\tD: {D}",
                        "idiom": idiom,
                        "A": A,
                        "B": B,
                        "C": C,
                        "D": D,
                        "target": target,
                    }
                )

    elif task_type == "idiom-extraction":
        with open(data_path, "r") as f:
            lines = [line.strip() for line in f.readlines()][1:]
            for line in lines:
                context, idiom = [item.strip() for item in line.split("\t")]
                data_examples.append({"context": context, "idiom": idiom})
    elif task_type in [
            "idiom-paraphrase",
            "sequential-task-idiom-extraction-and-interpretation"          
        ]:
        with open(data_path, "r") as f:
            lines = [line.strip() for line in f.readlines()][1:]
            for line in lines:
                id_, idiom, paraphrase, context_idiomatic, context_literal = [
                    item.strip() for item in line.split("\t")
                ]
                data_examples.append(
                    {
                        "id": id_,
                        "idiom": idiom,
                        "paraphrase": paraphrase,
                        "context_idiomatic": context_idiomatic,
                        "context_literal": context_literal,
                    }
                )
    elif task_type == "collocate-retrieval":
        raise NotImplementedError("Not implemented yet!")
    elif task_type in [
        "collocation-interpretation",
        "sequential-task-collocate-extraction-and-interpretation"
    ]:
        with open(data_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            for line in lines:
                if len(data_examples) >= max_num_limit:
                    break
                if line == "":
                    continue
                line = line.split("\t")
                if len(line) != 8:
                    continue
                colloction = line[3]
                context = line[6]
                paraphrases = line[7].split("#####")
                data_examples.append(
                    {
                        "collocation": colloction,
                        "context": context,
                        "paraphrases": paraphrases,
                    }
                )

    elif task_type in ["collocation-categorization", "collocation-scaling","sequential-task-collocate-extraction-and-judgment"]:
        with open(data_path, "r") as f:
            lines = [line.strip() for line in f.readlines()][1:]
            for line in lines:
                id_, base, collocate, collocation, label, label_id, context = [
                    item.strip() for item in line.split("\t")
                ]
                data_examples.append(
                    {
                        "id": id_,
                        "base": base,
                        "collocate": collocate,
                        "collocation": collocation,
                        "label": label,
                        "label_id": label_id,
                        "context": context,
                    }
                )

    elif task_type == "collocation-extraction":
        label2examples = generate_label_examples_map(
            data_path,
            max_num_limit,
            task_type,
        )
        data_examples = label2examples
    elif task_type in [
        "noun-compound-compositionality",
        "sequential-task-noun-compound-extraction-and-judgment"
    ]:
        with open(data_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            for line in lines:
                id_, noun_compound, context, A, B, C, D, target = [
                    item.strip() for item in line.split("\t")
                ]
                data_examples.append(
                    {
                        "id": id_,
                        "noun_compound": noun_compound,
                        "context": f"{context}\nOptions: A: {A}\tB: {B}\tC: {C}\tD: {D}",
                        "A": A,
                        "B": B,
                        "C": C,
                        "D": D,
                        "target": target,
                    }
                )
    elif task_type in [
        "noun-compound-interpretation",             
        ]:
        with open(data_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            for line in lines:
                _, noun_compound, references = line.split("\t")
                references = ast.literal_eval(references)
                data_examples.append(
                    {
                        "noun_compound": noun_compound,
                        "references": references,
                    }
                )
    elif task_type == "noun-compound-extraction":
        with open(data_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            for line in lines:
                context, _, _, noun_compound, _ = line.split("\t")
                data_examples.append(
                    {
                        "context": context,
                        "noun_compound": noun_compound,
                    }
                )
    elif task_type == "vmwe-extraction":
        with open(data_path, "r", encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            for line in lines:
                id_, context, vmwe, label = [item.strip() for item in line.split("\t")]
                data_examples.append(
                    {
                        "id": id_,
                        "context": context,
                        "vmwe": vmwe,
                        "label": label,
                    }
                )


    else:
        raise NotImplementedError(
            "Only support following list of tasks for now: [idiom-detection, idiom-extraction, idiom-paraphrase, collocate-retrieval, collocation-categorization, collocation-extraction, noun-compound-compositionality, noun-compound-interpretation, noun-compound-extraction"
        )

    return data_examples


def construct_prompt(
    args: Namespace,
    instance: Dict[str, Any],
    prompt_template: str,
    taxonomy: Optional[str] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    is_oracle: Optional[bool] = None,
) -> str:
    """
    Construct prompt from the given instance, prompt template, taxonomy, and examples

    Args:
    - args: arguments
    - instance: instance
    - prompt_template: prompt template
    - taxonomy: taxonomy
    - examples: examples

    Returns:
    - constructed prompt
    """

    def category2instances(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        category2instances = dict()
        for example in examples:
            label = example["label"].split(".")[0]
            if label not in category2instances:
                category2instances[label] = []
            category2instances[label].append(example)
        return category2instances

    prompt = ""
    # prepare prompt
    if taxonomy and "{{taxonomy}}" in prompt_template:
        prompt = prompt_template.replace("{{taxonomy}}", taxonomy)
    if "{{relation}}" in prompt_template:
        name2relation = generate_relation(instance["label"], args.taxonomy_path)
        if instance["label"] not in name2relation:
            raise ValueError(
                f"Label ``{instance['label']}'' not found in taxonomy {args.taxonomy_path}."
            )
        prompt = (
            prompt.replace("{{relation}}", name2relation[instance["label"]])
            if prompt != ""
            else prompt_template.replace(
                "{{relation}}", name2relation[instance["label"]]
            )
        )
    if examples and "{{example}}" in prompt_template and examples:
        if args.task == "idiom-detection":
            examples_str = "".join(
                [
                    f"\nIdiom: {example['idiom']}\nContext: {example['context']}\nOutput: {example['target']}\n"
                    for example in examples[: args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )
        elif args.task == "sequential-task-idiom-extraction-and-judgment":
            examples_str = "".join(
                [
                    f"\nContext: {example['context']}\nIdiom: {example['idiom']}\Option: {example['target']}\n"
                    for example in examples[: args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )
        elif args.task == "idiom-extraction":
            examples_str = "".join(
                [
                    f"\nContext: {example['context']}\nOutput: {example['idiom']}\n"
                    for example in examples[: args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )
        elif args.task == "idiom-paraphrase":
            examples_str = "".join(
                [
                    f"\nIdiom: {example['idiom']}\nContext: {example['context_idiomatic']}\nOutput: {example['paraphrase']}\n"
                    for example in examples[: args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )
        elif args.task == "sequential-task-idiom-extraction-and-interpretation":
            examples_str = "".join(
                [
                    f"\nContext: {example['context_idiomatic']}\nIdiom: {example['idiom']}\Paraphrase: {example['paraphrase']}\n"
                    for example in examples[: args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )

        elif args.task == "collocate-retrieval":
            raise NotImplementedError("Not implemented yet!")
        elif args.task in ["collocation-categorization", "collocation-scaling"]:
            examples_str = "".join(
                [
                    f"\nInput: {example['context']}; {example['collocation']}\nOutput: {example['label']}\n"
                    for example in examples[: args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )
        elif args.task == "sequential-task-collocate-extraction-and-judgment":
            examples_str = "".join(
                [
                    f"\nContext: {example['context']}\nCollocation:{example['collocation']}\nCategory: {example['label']}\n"
                    for example in examples[: args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )
        elif args.task == "collocation-extraction":
            label = instance["label"]
            examples_str = "".join(
                [f"{example} ; " for example in examples[label][: args.shot_num]]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )
        elif args.task == "collocation-interpretation":
            examples_str = "".join(
                [
                    f"\nCollocation: {example['collocation']}\nContext: {example['context']}\nInterpretation: {example['paraphrases'][0]}\n"
                    for example in examples[: args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )
        elif args.task == "sequential-task-collocate-extraction-and-interpretation":
            examples_str = "".join(
                [
                    f"\nContext: {example['context']}\nCollocation: {example['collocation']}\nParaphrases: {example['paraphrases'][0]}\n"
                    for example in examples[: args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )
            
        elif args.task == "noun-compound-compositionality":
            examples_str = "".join(
                [
                    f"\nNoun compound: {example['noun_compound']}\nContext: {example['context']}\nOutput: {example['target']}\n"
                    for example in examples[: args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )
        elif args.task == "sequential-task-noun-compound-extraction-and-judgment":
            examples_str = "".join(
                [
                    f"\nContext: {example['context']}\nNoun compound: {example['noun_compound']}\nOption: {example['target']}\n"
                    for example in examples[: args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )
        elif args.task == "noun-compound-extraction":
            examples_str = "".join(
                [
                    f"\nContext: {example['context']}\nNoun compound: {example['noun_compound']}\n"
                    for example in examples[: args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )
        elif args.task == "noun-compound-interpretation":
            examples_str = "".join(
                [
                    f"\nNoun compound: {example['noun_compound']}\nOutput: {example['references'][0]}\n"
                    for example in examples[: args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )
        elif args.task == "vmwe-extraction":
            if is_oracle:
                category2examples = category2instances(examples)
                label_prefix = instance["label"].split(".")[0]
                if label_prefix not in category2examples:
                    raise ValueError(
                        f"Label {instance['label']} not found in examples! Please check your examples for demonstration."
                    )
                examples = category2examples[label_prefix]
            random.shuffle(examples)
            examples_str = "".join(
                [
                    f"\nContext: {example['context']}\nExtracted VMWE (phrase only): {example['vmwe']}\n"
                    for example in examples[: args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    examples_str,
                )
            )
        
        elif args.task in ["sequential-task-idiom"]:
            examples_str = "".join(
                [
                    f"\nContext: {example['context_idiomatic']}\nExtracted Idiom: {example['idiom']}\nInterpretation: {example['paraphrase']}\n" for example in examples[:args.shot_num]
                ]
            )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    examples_str
                )
                if prompt == ""
                else prompt.place(
                    "{{example}}",
                    examples_str
                )
            )


        else:
            raise NotImplementedError("Not implemented yet!")
    if "{{context}}" in prompt_template:
        if args.task in [
            "idiom-paraphrase",
            "sequential-task-idiom-extraction-and-interpretation"
        ]:
            prompt = (
                prompt_template.replace("{{context}}", instance["context_idiomatic"])
                if prompt == ""
                else prompt.replace("{{context}}", instance["context_idiomatic"])
            )
        else:
            prompt = (
                prompt_template.replace("{{context}}", instance["context"])
                if prompt == ""
                else prompt.replace("{{context}}", instance["context"])
            )
    if "{{idiom}}" in prompt_template:
        prompt = (
            prompt.replace("{{idiom}}", instance["idiom"])
            if prompt != ""
            else prompt_template.replace("{{idiom}}", instance["idiom"])
        )
    if "{{collocation}}" in prompt_template:
        prompt = (
            prompt.replace("{{collocation}}", instance["collocation"])
            if prompt != ""
            else prompt_template.replace("{{collocation}}", instance["collocation"])
        )
    if "{{relation}}" in prompt_template:
        name2relation = generate_relation(instance["label"], args.taxonomy_path)
        if instance["label"] not in name2relation:
            raise ValueError(
                f"Label {instance['label']} not found in taxonomy {args.taxonomy_path}"
            )
        prompt = (
            prompt.replace("{{relation}}", name2relation[instance["label"]])
            if prompt != ""
            else prompt_template.replace(
                "{{relation}}", name2relation[instance["label"]]
            )
        )
    if "{{noun compound}}" in prompt_template:
        prompt = (
            prompt.replace("{{noun compound}}", instance["noun_compound"])
            if prompt != ""
            else prompt_template.replace("{{noun compound}}", instance["noun_compound"])
        )
    if "{{vmwe-definition}}" in prompt_template:
        prompt = (
            prompt.replace("{{vmwe-definition}}", instance["oracle"])
            if prompt != ""
            else prompt_template.replace("{{vmwe-definition}}", instance["oracle"])
        )

    return prompt


def generate_relation(label: str, taxonomy_path: str) -> str:
    """
    Generate relation from label

    Args:
    - label: label
    - taxonomy_path: taxonomy path

    Returns:
    - look-up table from class name to relation
    """
    name2relation = dict()
    with open(taxonomy_path, "r") as f:
        for line in [l.strip() for l in f.readlines()]:
            if line == "":
                continue
            line = line.split("\t")
            name2relation[line[1].strip()] = line[3].strip()
    return name2relation


def generate_label_examples_map(
    data_path: str, max_num_limit: int, task: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Generate label to examples map

    Args:
    - data_examples: path pointed to data examples
    - max_num_limit: max number of data
    - task: task type

    Returns:
    - label2examples: label to examples map
    """
    label2examples = dict()
    with open(data_path, "r") as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            label, examples_str = line.strip().split("\t")
            examples = [e.strip() for e in examples_str.split(",")]
            label2examples[label.strip()] = examples[:max_num_limit]

    return label2examples


def postprocess(
    s: str, task: Optional[str] = None, args: Optional[Namespace] = None
) -> str:
    def match_option(s: str, pattern: str) -> str:
        if any(["A" + pattern in s, pattern + "A" in s]):
            return "A"
        if any(["B" + pattern in s, pattern + "B" in s]):
            return "B"
        if any(["C" + pattern in s, pattern + "C" in s]):
            return "C"
        if any(["D" + pattern in s, pattern + "D" in s]):
            return "D"

    s = " ".join(s.split())
    # s = re.sub(r"\[(.*?)\]", "", s)
    if task == "collocate-retrieval":
        if "1." in s:
            order, collocate = s.split("1.")
    if task == "collocation-categorization":
        if "is: " in s:
            s = s.split("is: ")[-1].strip()
        elif "Output:" in s:
            s = s.split("Output:")[-1].strip()
        return s
        
    if task == "idiom-detection":
        if "mixtral" in args.model.lower():
            if any("is [" + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("is [") + len("is [")].strip()
                return s
        probe_option = (
            "A"
            if any(s.startswith(p) for p in ["A ", "A: ", "A."]) or "option is A" in s
            else "B"
            if any(s.startswith(p) for p in ["B ", "B: ", "B."]) or "option is B" in s
            else "C"
            if any(s.startswith(p) for p in ["C ", "C: ", "C."]) or "option is C" in s
            else "D"
            if any(s.startswith(p) for p in ["D ", "D: ", "D."]) or "option is D" in s
            else s
        )
        if len(probe_option) == 1:
            s = probe_option
        else:
            s = (
                s.replace(":", "")
                .replace("(", "")
                .replace(")", "")
                .replace('"', "")
                .replace("'", "")
            )
            option2freq = dict()
            words = s.split()
            for word in words:
                if word in ["A", "B", "C", "D"]:
                    option2freq[word] = option2freq.get(word, 0) + 1
                if word in [
                    "A:",
                    "B:",
                    "C:",
                    "D:",
                    "A.",
                    "B.",
                    "C.",
                    "D.",
                    "A,",
                    "B,",
                    "C,",
                    "D,",
                    "A -",
                    "B -",
                    "C -",
                    "D -",
                ]:
                    option2freq[word[0]] = option2freq.get(word[0], 0) + 1
                if word in [
                    "- A",
                    "- B",
                    "- C",
                    "- D",
                    "option A",
                    "option B",
                    "option C",
                    "option D",
                ]:
                    option2freq[word.split()[-1]] = (
                        option2freq.get(word.split()[-1], 0) + 1
                    )
            return max(option2freq, key=option2freq.get) if len(option2freq) > 0 else s
    if task == "idiom-paraphrase":
        if ":" in s:
            s = s.split(":", maxsplit=1)[-1].strip()
        if "=" in s:
            s = s.split("=", maxsplit=1)[-1].strip()
        if ";" in s:
            s = s.split(";")[0].strip()
        if "be paraphrased as" in s:
            s = s.split("be paraphrased as")[-1].strip()
        if "in the given context" in s:
            s = s.split("in the given context")[-1].strip()
        if "in the given sentence" in s:
            s = s.split("in the given sentence")[-1].strip()
        if " means ":
            s = s.split(" means ")[-1].strip()
        if "refers to":
            s = s.split("refers to")[-1].strip()
    if task == "idiom-extraction":
        if "Extracted Idiom:" in s:
            s = s.split("Extracted Idiom:")[-1].strip()
    if task == "noun-compound-compositionality":
        # heurustic rules for proprietary gpt-like models
        if any(p in args.model.lower() for p in ["gpt", "gemini", "claude"]):
            patterns = re.findall(r'"(.*?)"', s)
            if "Output Option (Option only, no analysis):" in s:
                s = s.split("Output Option (Option only, no analysis):")[-1].strip()
            if any("Output: " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s.split("Output: ")[-1].strip()
            elif any("Output:" + opt in s for opt in ["A", "B", "C", "D"]):
                s = s.split("Output:")[-1].strip()
            elif any(opt + ":" in s for opt in ["A", "B", "C", "D"]):
                s = s.rsplit(":", maxsplit=1)[0].strip()
            if any(
                opt == s.split(" ", maxsplit=1)[0].strip()
                for opt in ["A", "B", "C", "D"]
            ):
                s = s.split(" ", maxsplit=1)[0].strip()
            if any(opt + "." in s for opt in ["A", "B", "C", "D"]):
                s = s.split(".")[0].strip()
            if any("option is " + opt in s for opt in ["A", "B", "C", "D"]):
                # get index of this pattern and move to next word
                s = s[s.index("option is ") + len("option is ")]
            if any("Option is " + opt in s for opt in ["A", "B", "C", "D"]):
                # get index of this pattern and move to next word
                s = s[s.index("Option is ") + len("Option is ")]
            if any("is option " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("is option ") + len("is option ")]
            if any("is Option " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("is Option ") + len("is Option ")]
            if any("in this context is " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("in this context is ") + len("in this context is ")]
            if any("in this context is: " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("in this context is: ") + len("in this context is: ")]
            if any("answer is " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("answer is ") + len("answer is ")]
            if any("answer is: " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("answer is: ") + len("answer is: ")]
            if patterns:
                for pattern in patterns:
                    if (
                        "compositional" in pattern.lower()
                        or "none of the above" in pattern.lower()
                    ):
                        s = pattern
            if any(opt + " " in s for opt in ["A", "B", "C", "D"]):
                s = s.split(" ")[0].strip()
            elif any("is: " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s.split("is: ")[-1].strip()
        # heurustic rules for llama-2 models
        elif any(p in args.model.lower() for p in ["llama", "mistral", "mixtral"]):
            if any("option is: " + opt in s for opt in ["A", "B", "C", "D"]):
                # get index of this pattern and move to next word
                s = s[s.index("option is: ") + len("option is: ")].strip()
            elif any("Option is: " + opt in s for opt in ["A", "B", "C", "D"]):
                # get index of this pattern and move to next word
                s = s[s.index("Option is: ") + len("Option is: ")].strip()
            if any("option is " + opt in s for opt in ["A", "B", "C", "D"]):
                # get index of this pattern and move to next word
                s = s[s.index("option is ") + len("option is ")].strip()
            elif any("Option is " + opt in s for opt in ["A", "B", "C", "D"]):
                # get index of this pattern and move to next word
                s = s[s.index("Option is ") + len("Option is ")].strip()
            if any("is option " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("is option ") + len("is option ")].strip()
            elif any("is Option " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("is Option ") + len("is Option ")].strip()
            if any("is option: " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("is option: ") + len("is option: ")].strip()
            elif any("is Option: " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("is Option: ") + len("is Option: ")].strip()
            if any("Option: " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("Option: ") + len("Option: ")].strip()
            if any("in this context is " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[
                    s.index("in this context is ") + len("in this context is ")
                ].strip()
            if any("in this context is: " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[
                    s.index("in this context is: ") + len("in this context is: ")
                ].strip()
            if any("answer is " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("answer is ") + len("answer is ")].strip()
            if any("answer is: " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("answer is: ") + len("answer is: ")].strip()
            if any("compound as:" + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("compound as:") + len("compound as:")].strip()
            elif any("compound as " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("compound as ") + len("compound as ")].strip()
            if any("compound as " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("compound as ") + len("compound as ")].strip()
            if any("as Option " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("as Option ") + len("as Option ")].strip()
            if any("classified as: " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("classified as: ") + len("classified as: ")].strip()
            elif any("classified as " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("classified as ") + len("classified as ")].strip()
            if any("described as: " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("described as: ") + len("described as: ")].strip()
            elif any("described as " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("described as ") + len("described as ")].strip()
            if any(" as: " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index(" as: ") + len(" as: ")].strip()
            if any(" as " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index(" as ") + len(" as ")].strip()
            if any("as: " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("as: ") + len("as: ")].strip()
            elif any("as " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("as ") + len("as ")].strip()
            if any(opt + ":" in s for opt in ["A", "B", "C", "D"]):
                s = match_option(s, ":")
            if any("I would choose option " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[
                    s.index("I would choose option ") + len("I would choose option ")
                ].strip()
            elif any(
                "I would choose Option " + opt in s for opt in ["A", "B", "C", "D"]
            ):
                s = s[
                    s.index("I would choose Option ") + len("I would choose Option ")
                ].strip()
            if any(" is (" + opt in s for opt in ["A", "B", "C", "D"]):
                s = s.split(" is (")[-1].split(")")[0].strip()
            if any("option (" + opt in s for opt in ["A", "B", "C", "D"]):
                s = s.split("option (")[-1].split(")")[0].strip()
            if any("Option (" + opt in s for opt in ["A", "B", "C", "D"]):
                s = s.split("Option (")[-1].split(")")[0].strip()
            if "(Option " in s:
                s = s[s.index("(Option ") + len("(Option ")]
            elif "option is " in s:
                # match = re.search(r"\(([A-D])\)|\b([A-D])\b", s)
                # if match:
                # s = match.group(1) or match.group(2)
                s = s[s.index("option is ") + len("option is ")].strip()
            if "in the given sentence is " in s:
                s = s[
                    s.index("in the given sentence is ")
                    + len("in the given sentence is ")
                ].strip()
            elif any(" option " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index(" option ") + len(" option ")].strip()
            if any("(" + opt in s for opt in ["A", "B", "C", "D"]):
                s = s.split("(")[-1].split(")")[0].strip()
            if any(" is " + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index(" is ") + len(" is ")].strip()
            if any("answer is:" + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("answer is:") + len("answer is:")].strip()
            elif any(
                "appropriate option is:" + opt in s for opt in ["A", "B", "C", "D"]
            ):
                s = s[
                    s.index("appropriate option is:") + len("appropriate option is:")
                ].strip()
            if any("is:" + opt in s for opt in ["A", "B", "C", "D"]):
                s = s[s.index("is:") + len("is:")].strip()
            s = s.replace(",", "")
            s = s.replace(".", "")
            if "this compound as" in s:
                s = s.split("this compound as")[-1].strip()
            elif "the compound as" in s:
                s = s.split("the compound as")[-1].strip()
    if task == "noun-compound-interpretation":
        if "Sure, I'd be happy to help!":
            s = s.replace("Sure, I'd be happy to help!", "").strip()
        elif "Sure, I can help you with that!":
            s = s.replace("Sure, I can help you with that!", "").strip()
        if "I would paraphrase it as":
            s = s.split("I would paraphrase it as")[-1].strip()
        if "refers to":
            s = s.split("refers to")[-1].strip()
        if "paraphrased as":
            s = s.split("paraphrased as")[-1].strip()
        if "I would paraphrase":
            s = s.split("I would paraphrase")[-1].strip()
        if "can be paraphrased as":
            s = s.split("can be paraphrased as")[-1].strip()
        if ":" in s:
            s = s.split(":")[-1].strip()
    if task == "noun-compound-extraction":
        if "A noun compound" in s:
            s = s.split("A noun compound")[-1].strip()
        if "Noun compound:" in s:
            s = s.split("Noun compound:")[-1].strip()

    if task == "sequential-task-collocate-extraction-and-judgment":
        result = {
            "extracted_collocation": None,
            "category": None
        }
        extraction_pattern = r"Extracted Collocation:\s*(.+?)(?=\s*Category:|$)"
        category_pattern = r"Category:\s*(.+)"

        extraction_match = re.search(extraction_pattern, s, re.IGNORECASE)
        category_match = re.search(category_pattern, s, re.IGNORECASE)

        if extraction_match:
            extracted_text = extraction_match.group(1).strip()
            extracted_text = re.sub(r'\n.*', '', extracted_text)
            result["extracted_collocation"] = extracted_text
           
    
        if category_match:
            category_text = category_match.group(1).strip()
            category_text = re.sub(r'\n.*', '', category_text)
            result["category"] = category_text
            
        return result
    
    if task == "sequential-task-idiom-extraction-and-judgment":
        s = s.replace("\r", "").strip()

        result = {
            "extracted_idiom": None,
            "option": None,
        }

        extraction_pattern = r"Extracted Idiom:\s*(.+?)(?=\s*Option:|$)"
        option_pattern = r"Option:\s*([A-D])"

        extraction_match = re.search(extraction_pattern, s, re.IGNORECASE)
        option_match = re.search(option_pattern, s, re.IGNORECASE)

        if extraction_match:
            idiom_text = extraction_match.group(1).strip()
            idiom_text = re.sub(r'\n.*', '', idiom_text)
            result["extracted_idiom"] = idiom_text

        if option_match:
            option_text = option_match.group(1).strip().upper()
            result["option"] = option_text

        return result
    
    if task == "sequential-task-noun-compound-extraction-and-judgment":
        result = {
            "extracted_collocation": None,
            "category": None
        }

        s = s.replace("\r", "")

        extraction_pattern = r"Extracted Noun Compound:\s*(.+?)(?=\s*Selected Option:|$)"
        category_pattern = r"Selected Option:\s*(.+)"

        extraction_match = re.search(extraction_pattern, s, re.IGNORECASE)
        category_match = re.search(category_pattern, s, re.IGNORECASE)

        if extraction_match:
            extracted = extraction_match.group(1).strip()
            extracted = re.sub(r"\n.*", "", extracted)
            result["extracted_collocation"] = extracted

        if category_match:
            category = category_match.group(1).strip()
            category = re.sub(r"\n.*", "", category)
            result["category"] = category

        return result

    
    if task == "sequential-task-idiom-extraction-and-interpretation":
        result = {
            "extracted_idiom": None,
            "paraphrase": None
        }

        s = s.replace("\r\n", "\n").replace("\r", "\n")



        extraction_pattern = r"Extracted Idiom:\s*(.+?)(?=\s*Paraphrase:|$)"
        paraphrase_pattern = r"Paraphrase:\s*(.+)"

        extraction_match = re.search(extraction_pattern, s, re.IGNORECASE)
        paraphrase_match = re.search(paraphrase_pattern, s, re.IGNORECASE)
        
        if extraction_match:
            extracted_idiom = extraction_match.group(1).strip()
            extracted_idiom = re.sub(r"\n.*", "", extracted_idiom)
            result["extracted_idiom"] = extracted_idiom

        if paraphrase_match:
            paraphrase = paraphrase_match.group(1).strip()
            paraphrase = re.sub(r"\n.*", "", paraphrase)
            result["paraphrase"] = paraphrase

        return result

    if task == "sequential-task-collocate-extraction-and-interpretation":
        result = {
            "extracted_collocation": None,
            "paraphrase": None
        }

        s = s.replace("\r", "")

        collocation_pattern = r"Extracted Collocation:\s*(.+?)(?=\s*Paraphrase:|$)"
        paraphrase_pattern = r"Paraphrase:\s*(.+)"

        collocation_match = re.search(collocation_pattern, s, re.IGNORECASE)
        paraphrase_match = re.search(paraphrase_pattern, s, re.IGNORECASE)

        if collocation_match:
            extracted_collocation = collocation_match.group(1).strip()
            extracted_collocation = re.sub(r"\n.*", "", extracted_collocation)
            result["extracted_collocation"] = extracted_collocation

        if paraphrase_match:
            paraphrase = paraphrase_match.group(1).strip()
            paraphrase = re.sub(r"\n.*", "", paraphrase)
            result["paraphrase"] = paraphrase

        return result

    
    if task == "vmwe-extraction":
        if "llama" in args.model.lower():
            matched_patterns = re.findall('"(.*?)"', s)
            if matched_patterns:
                s = max(matched_patterns, key=len)
        s = s.replace("*", "").strip()
        if "This is a" in s:
            s = s.split("This is a")[0].strip()
        if "This phrase is" in s:
            s = s.split("This phrase is")[0].strip()
        if "It is a" in s:
            s = s.split("It is a")[0].strip()
        if "is a verbal multiword expression" in s:
            s = (
                s.split("is a verbal multiword expression")[0]
                .strip()
                .split(",")[-1]
                .strip()
            )
        if "This VMWE" in s:
            s = s.split("This VMWE")[0].strip()
        if "in the given sentence is" in s:
            s = s.split("in the given sentence is")[-1].strip()
        if "in the sentence is" in s:
            s = s.split("in the sentence is")[-1].strip()
        if "in this sentence is" in s:
            s = s.split("in this sentence is")[-1].strip().split(",")[0].strip()
        if "This expression is" in s:
            s = s.split("This expression is")[0].strip()
        if "This phrase consists of" in s:
            s = s.split("This phrase consists of")[0].strip()
        if "verbal multiword expression is" in s:
            s = s.split("verbal multiword expression is")[-1].strip()
        if "is a VMWE" in s:
            s = s.split("is a VMWE")[0].strip()
        if "is a common idiomatic expression" in s:
            s = s.split("is a common idiomatic expression")[0].strip()
        if "Verbal Multiword Expression (VMWE) is" in s:
            s = s.split("Verbal Multiword Expression (VMWE) is")[-1].strip()
        if "Verbal Multiword Expression (VMWE):" in s:
            s = s.split("Verbal Multiword Expression (VMWE):")[-1].strip()
    else:
        if "Output:" in s:
            s = s.split("Output:")[-1].strip()
        elif "output:" in s:
            s = s.split("output:")[-1].strip()
        elif "Prediction:" in s:
            s = s.split("Prediction:")[-1].strip()
        elif "prediction:" in s:
            s = s.split("prediction:")[-1].strip()
        elif "Predict:" in s:
            s = s.split("Predict:")[-1].strip()
        elif "predict:" in s:
            s = s.split("predict:")[-1].strip()
        elif "Paraphrase:" in s:
            s = s.split("Paraphrase:")[-1].strip()
        elif "paraphrase:" in s:
            s = s.split("paraphrase:")[-1].strip()

        s = s.replace('"', "").strip()
        s = s.replace(".", "").strip()

    



    return s.strip()