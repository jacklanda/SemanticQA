# -*- coding: utf-8 -*-

import json
import warnings
from typing import Dict, List, Generator, Callable, Optional

import tiktoken
from rich.console import Console


class DeprecatedError(Exception):
    pass


def deprecated(func: Callable, warning: bool = False):
    """Decorator for deprecated functions.

    Args:
    - func: the function to be deprecated.
    - warning: whether to show warning or raise error.

    Returns:
    - wrapper: the wrapper function.
    """

    def wrapper(*args, **kwargs):
        """Wrapper function for deprecated functions."""
        message = f"The function {func.__name__} is deprecated and should not be used."

        if warning:
            warnings.warn(message, DeprecationWarning)
        else:
            raise DeprecatedError(message)

        return func(*args, **kwargs)

    return wrapper


def load_prompt(prompt_path: str):
    """Load prompt from file.
    :param prompt_path: path to the prompt file.
    :return: prompt.
    """
    with open(prompt_path, "r", encoding='utf-8') as f:
        prompt = f.read()
    return prompt


def load_taxonomy(taxonomy_path: str, args: Optional[str] = None):
    """Load taxonomy from file.
    :param taxonomy_path: path to the taxonomy file.
    :return: taxonomy.
    """
    with open(taxonomy_path, "r") as f:
        lines = f.readlines()
    # it should not provide demos for few-shot setting
    taxonomy = ""
    for line in lines:
        line_items = line.split("\t")
        if args.shot_num > 0:
            taxonomy += line
        elif args.shot_num == 0 and len(line_items) == 2:  # Label \t Semantic Gloss
            line_wo_demos = line_items[0] + "\t" + line_items[1]
            taxonomy += line_wo_demos
        elif args.shot_num == 0 and len(line_items) == 3:
            line_wo_demos = line_items[0] + "\t" + line_items[2]
            taxonomy += line_wo_demos
        elif args.shot_num == 0 and len(line_items) == 4:  # ID \t Label \t Examples \t Semantic Gloss
            taxonomy += line_items[1] + "\t" + line_items[3]
    taxonomy = taxonomy.replace("Semantic Gloss", "Meaning")
    return taxonomy


def load_tokenizer(model: str) -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(model)


def count_token(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a string.
    :param s: string.
    :return: number of tokens.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def load_json(file_path: str) -> Generator:
    """Load json from file.
    :param file_path: path to the json file.
    :return: json.
    """
    with open(file_path, "r") as f:
        for line in f.readlines():
            yield json.loads(line)


def dump_json(
    output_path: str, json_list: List[Dict[str, str]], indent: bool = False
) -> None:
    """Dump a list of json to a file.
    :param output_path: path to the output file.
    :param json_list: a list of json.
    :param indent: whether to indent the json.
    :return: None.
    """
    with open(output_path, "w", encoding='utf-8') as f:
        for line in json_list:
            if indent:
                f.write(json.dumps(line, ensure_ascii=False, indent=4) + "\n")
            else:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")


def dump_tsv(
    output_path: str, json_list: List[Dict[str, str]], sep: str = "\t"
) -> None:
    """Dump a list of json to a tsv file.
    :param output_path: path to the output file.
    :param json_list: a list of json.
    :param sep: separator.
    :return: None.
    """
    with open(output_path, "w", encoding='utf-8') as f:
        for json_obj in json_list:
            f.write(sep.join(json_obj.values()) + "\n")


def compute_metric(pred: str, gold: str) -> float:
    """Compute the metric between the prediction and the gold.
    :param pred: prediction.
    :param gold: gold.
    :return: metric.
    """
    return 0.0


if __name__ == "__main__":
    console = Console()

    # test `load_prompt()`
    console.print("test `load_prompt()`")
    prompt = load_prompt(prompt_file_path="prompts/LC_REC_ZS.txt")
    console.print(prompt)

    # test `load_taxonomy()`
    console.print("test `load_taxonomy()`")
    taxonomy = load_taxonomy(taxonomy_file_path="taxonomy/SEM_REL_CATEGORY.txt")
    console.print(taxonomy)

    # test `count_token()`
    console.print("test `count_token()`")
    s = """- Magn: Intense, strong degree, an intensifier of semantic relation for base lexeme. Intensify the base lexeme to a high level, strengthening its semantic relation with the associated concept via the collocate lexeme.
    - AntiMagn: Slight and weak degree, a de-intensifier, Weaken meaning intensity, diminishing the semantic relationship between the base lexeme and its associated concept.
    - Ver: Lat. verus, real, genuine, As it should be, Meet intended requirements.
    - AntiVer: Non-genuine, Characterize something as non-genuine, not authentic, not in its intended or proper state, and not meeting the required standards or expectations.
    - IncepPredPlus: Start to increase., Denote the initiation of a process or action that leads to an increase or enhancement of something.
    - FinFunc0: End.existence, The value means "the base word of FinFunc0 ceases to be experienced".
    - Fact0: Lat. factum, fact. To fulfill the requirement of base word, and the argument of this function fulfills its own requirement., Fulfill the requirement of base word, do something with base word, what you are supposed to do with base word.
    - CausFunc0: The agent does something so that the event denoted by the noun occurs, Do something so that base word begins occurring.
    - CausFact0: To cause something to function according to its destination., Denote causing something to function according to its intended purpose or destination.
    - CausPredMinus: Cause to decrease., Describe the act of causing a decrease or reduction in something.
    - CausFunc1: The non-agentive participant does something such that the event denoted by the noun occurs., A person/object, different from the agent of base word, does something so that base word occurs and has effect on the agent of base word.
    - Son: Lat. sonare: sound., The base word is usually a noun, and the value means "emit a characteristic sound".
    - Oper1: Lat. operari: perform, do, act something., Represent a light verb linking the event's first participant (subject) with the event's name (direct object).
    - IncepOper1: Incep is from Lat. incipere: begin. Begin to do, perform, experience, carry out base word., Signify the start of an action or event, linking the event's subject with its name using a light verb.
    - FinOper1: Fin is from Lat. finire: cease., Terminate doing something.
    - Real1: Fulfill a requirement imposed by the noun or performing an action typical for the noun., To fulfill the requirement of base word, to act according to base word.
    - Real2: Acting as expected. Something be realized as expected, Do with regard to A that which is normally expected of second participant.
    - AntiReal2: Not acting as expected. Something not be realized as expected., The value is negation of an internal element of the argument of this function.
    """
    toks = count_token(s)
    console.print(f"Number of tokens: {toks}")
