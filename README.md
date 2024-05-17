# LexBench
[📄 **Paper**](https://arxiv.org/abs/2405.02861) **·** [📚 **Dataset**](https://github.com/jacklanda/LexBench/tree/main/lexbench/dataset) **·** [💻 **Code**](https://github.com/jacklanda/LexBench/tree/main/lexbench)

The official repository of the research paper ``Revisiting a Pain in the Neck: Semantic Phrase Processing Benchmark for Language Models''.

## Prepare Environments
```bash
conda env create -f environment.yml

cd lexbench
pip install -r requirements.txt
```

## Prepare Data
```bash
unzip resources/dataset.zip -d lexbench/
```

## Running Evaluation on Specific Task 
For example, the command for running idiom interpretation with `Claude-3-opus` is shown below.

```bash
python main.py \
  --task idiom-paraphrase \
  --api_key <Your API key> \
  --model claude-3-opus-20240229 \
  --prompt_path prompts/idiom_paraphrase_zeroshot.txt \
  --example_path dataset/idiom_paraphrase/prepared/examples.tsv \
  --input_path dataset/idiom_paraphrase/prepared/idiom_paraphrase_prepared.tsv \
  --output_path results/idiom-paraphrase_0-shot_claude-instant-1.json \
  --evaluate \
  --shot_num 0 \
  --max_query 1000 \
  --max_tokens 128 \
  --temperature 0 \
  --presence_penalty 0 \
  --frequency_penalty 0
```

## Benchmarking Scaling-category Semantic Categorization

```bash
./run_lcc_scaling.sh
```

## Citing LexBench
If you use LexBench's data or code in your research, please use the following BibTeX entry.
```latex
@article{liu2024revisiting,
  title={Revisiting a Pain in the Neck: Semantic Phrase Processing Benchmark for Language Models},
  author={Liu, Yang and Qin, Melissa Xiaohui and Li, Hongming and Huang, Chao},
  journal={arXiv preprint arXiv:2405.02861},
  year={2024}
}
```
