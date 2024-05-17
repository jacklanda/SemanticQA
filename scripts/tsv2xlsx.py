# -*- coding: utf-8 -*-
#
# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2024/01/10

import json

import pandas as pd


def dump2jsonl(input_df, output_file):
    with open(output_file, "w") as f:
        for row in input_df.iterrows():
            json.dump(row[1].to_dict(), f)
            f.write("\n")


def tsv_to_xlsx_idiom_detection(input_file, output_file):
    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(input_file, sep="\t")

    # Sample 100 rows from the DataFrame
    sampled_df = df.sample(
        n=100, random_state=42
    )  # Adjust random_state as needed for reproducibility

    # Write the sampled DataFrame to an XLSX file
    sampled_df.to_excel(output_file, index=False)


def tsv_to_xlsx_idiom_extraction(input_file, output_file):
    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(input_file, sep="\t")

    # Sample 100 rows from the DataFrame
    sampled_df = df.sample(
        n=100, random_state=42
    )  # Adjust random_state as needed for reproducibility

    # Write the sampled DataFrame to an XLSX file
    sampled_df.to_excel(output_file, index=False)


def tsv_to_xlsx_idiom_interpretation(input_file, output_file):
    # Read the TSV file into a pandas DataFrame
    df_all = pd.read_csv(input_file, sep="\t")

    # Sample 100 rows from the DataFrame
    sampled_df_all = df_all.sample(
        n=100, random_state=42
    )  # Adjust random_state as needed for reproducibility

    # keep only the three columns "idiom", "paraphrase", "context_idiomatic"
    df = sampled_df_all[["idiom", "paraphrase", "context_idiomatic"]]

    # change column name "context_idiomatic" to "context"
    sampled_df = df.rename(columns={"context_idiomatic": "context"})

    # Write the sampled DataFrame to an XLSX file
    dump2jsonl(sampled_df_all, output_file.replace("xlsx", "json"))
    sampled_df.to_excel(output_file, index=False)


def tsv_to_xlsx_collocation_categorization(input_file, output_file):
    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(input_file, sep="\t")

    # only keep the 4th 5th, 7th columns
    df = df.iloc[:, [3, 4, 6]]

    # Sample 100 rows from the DataFrame
    sampled_df = df.sample(
        n=100, random_state=42
    )  # Adjust random_state as needed for reproducibility

    # Write the sampled DataFrame to an XLSX file
    sampled_df.to_excel(output_file, index=False)


def tsv_to_xlsx_collocation_extraction(input_file, output_file):
    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(input_file, sep="\t")

    # only keep the 4th 5th, 7th columns
    df = df.iloc[:, [3, 4, 6]]

    # Sample 100 rows from the DataFrame
    sampled_df = df.sample(
        n=100, random_state=42
    )  # Adjust random_state as needed for reproducibility

    # Write the sampled DataFrame to an XLSX file
    sampled_df.to_excel(output_file, index=False)


def tsv_to_xlsx_collocation_extraction(input_file, output_file):
    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(input_file, sep="\t")

    # only keep the 4th 5th, 7th columns
    df = df.iloc[:, [3, 6]]

    # Sample 100 rows from the DataFrame
    sampled_df = df.sample(
        n=100, random_state=42
    )  # Adjust random_state as needed for reproducibility

    # Write the sampled DataFrame to an XLSX file
    sampled_df.to_excel(output_file, index=False)


def tsv_to_xlsx_collocation_interpretation(input_file, output_file):
    # Read the TSV file into a pandas DataFrame
    df_all = pd.read_csv(input_file, sep="\t")

    # Sample 100 rows from the DataFrame
    sampled_df_all = df_all.sample(
        n=100, random_state=42
    )  # Adjust random_state as needed for reproducibility

    # only keep the 4th 5th, 7th columns
    sampled_df = sampled_df_all.iloc[:, [6, 3, 7]]

    # Write the sampled DataFrame to an XLSX file
    dump2jsonl(sampled_df_all, output_file.replace("xlsx", "json"))
    sampled_df.to_excel(output_file, index=False)


def tsv_to_xlsx_noun_compound_compositionality(input_file, output_file):
    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(input_file, sep="\t")

    # only keep the 4th 5th, 7th columns
    df = df.iloc[:, [2, 1, 3, 4, 5, 6, 7]]

    # Sample 100 rows from the DataFrame
    sampled_df = df.sample(
        n=100, random_state=42
    )  # Adjust random_state as needed for reproducibility

    # Write the sampled DataFrame to an XLSX file
    sampled_df.to_excel(output_file, index=False)


def tsv_to_xlsx_noun_compound_extraction(input_file, output_file):
    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(input_file, sep="\t")

    # only keep the 4th 5th, 7th columns
    df = df.iloc[:, [0, 3]]

    # Sample 100 rows from the DataFrame
    sampled_df = df.sample(
        n=100, random_state=42
    )  # Adjust random_state as needed for reproducibility

    # Write the sampled DataFrame to an XLSX file
    sampled_df.to_excel(output_file, index=False)


def tsv_to_xlsx_noun_compound_interpretation(input_file, output_file):
    # Read the TSV file into a pandas DataFrame
    df_all = pd.read_csv(input_file, sep="\t")

    # Sample 100 rows from the DataFrame
    sampled_df_all = df_all.sample(
        n=100, random_state=42
    )  # Adjust random_state as needed for reproducibility

    # only keep the 4th 5th, 7th columns
    sampled_df = df_all.iloc[:, [1, 2]]

    # Write the sampled DataFrame to an XLSX file
    dump2jsonl(sampled_df_all, output_file.replace("xlsx", "json"))
    sampled_df.to_excel(output_file, index=False)


def tsv_to_xlsx_vmwe_extraction(input_file, output_file):
    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(input_file, sep="\t")

    # only keep the 4th 5th, 7th columns
    df = df.iloc[:, [1, 2, 3]]

    # Sample 100 rows from the DataFrame
    sampled_df = df.sample(
        n=100, random_state=42
    )  # Adjust random_state as needed for reproducibility

    # Write the sampled DataFrame to an XLSX file
    sampled_df.to_excel(output_file, index=False)


if __name__ == "__main__":
    # input_file = "dataset/idiom_detection/prepared/idiom_detection_prepared.tsv"
    # output_file = "dataset/idiom_detection/prepared/idiom_detection_sampled.xlsx"
    # tsv_to_xlsx_idiom_detection(input_file, output_file)

    # input_file = "dataset/idiom_extraction/prepared/idiom_extraction_prepared.tsv"
    # output_file = "dataset/idiom_extraction/prepared/idiom_extraction_prepared.xlsx"
    # tsv_to_xlsx_idiom_extraction(input_file, output_file)

    input_file = "dataset/idiom_paraphrase/prepared/idiom_paraphrase_prepared.tsv"
    output_file = "dataset/idiom_paraphrase/prepared/idiom_paraphrase_prepared.xlsx"
    tsv_to_xlsx_idiom_interpretation(input_file, output_file)

    # input_file = "dataset/collocation_categorization/prepared/collocation_categorization_prepared.tsv"
    # output_file = "dataset/collocation_categorization/prepared/collocation_categorization_prepared.xlsx"
    # tsv_to_xlsx_collocation_categorization(input_file, output_file)

    # input_file = (
    # "dataset/collocation_extraction/prepared/collocation_extraction_prepared.tsv"
    # )
    # output_file = (
    # "dataset/collocation_extraction/prepared/collocation_extraction_prepared.xlsx"
    # )
    # tsv_to_xlsx_collocation_extraction(input_file, output_file)

    # input_file = "dataset/collocation_paraphrase/collocation_paraphrase_prepared.tsv"
    # output_file = "dataset/collocation_paraphrase/collocation_paraphrase_prepared.xlsx"
    # tsv_to_xlsx_collocation_interpretation(input_file, output_file)

    # input_file = "dataset/noun_compound_compositionality/prepared/noun_compound_compositionality_prepared.tsv"
    # output_file = "dataset/noun_compound_compositionality/prepared/noun_compound_compositionality_prepared.xlsx"
    # tsv_to_xlsx_noun_compound_compositionality(input_file, output_file)

    # input_file = "dataset/noun_compound_extraction/prepared/noun_compound_extraction_prepared.tsv"
    # output_file = "dataset/noun_compound_extraction/prepared/noun_compound_extraction_prepared.xlsx"
    # tsv_to_xlsx_noun_compound_extraction(input_file, output_file)

    # input_file = "dataset/noun_compound_interpretation/prepared/noun_compound_interpretation_prepared.tsv"
    # output_file = "dataset/noun_compound_interpretation/prepared/noun_compound_interpretation_prepared.xlsx"
    # tsv_to_xlsx_noun_compound_interpretation(input_file, output_file)

    # input_file = (
    # "dataset/verbal_mwe_extraction/prepared/vmwe_identification_unique_prepared.tsv"
    # )
    # output_file = "dataset/verbal_mwe_extraction/prepared/vmwe_identification_unique_prepared.xlsx"
    # tsv_to_xlsx_vmwe_extraction(input_file, output_file)
