# SemanticQA

<p align="center">
  <a href="https://semanticqa.github.io">Project Page</a> •
  <a href="https://arxiv.org/abs/2604.16593">Paper (arXiv)</a> •
  <a href="https://huggingface.co/datasets/jacklanda/SemanticQA">Data (Hugging Face)</a> •
  <a href="https://github.com/jacklanda/SemanticQA">Evals (GitHub)</a> •
  <a href="https://github.com/jacklanda/SemanticQA/blob/main/slides.pdf">Slides</a> •
  <a href="https://github.com/jacklanda/SemanticQA/blob/main/poster.pdf">Poster</a>
</p>

The repository of the research project ``Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models''.

## Supported Tasks

SemanticQA supports 19 tasks across four phrase types, including 6 sequential (multi-step) tasks.

| Task                                   | Abbr. | Eval Metrics                                    | Phrase Type   |
|----------------------------------------|-------|-------------------------------------------------|---------------|
| Idiomatic Expression Detection         | IED   | MCQ Accuracy                                    | Idiom         |
| Idiomatic Expression Extraction        | IEE   | Exact Match                                     | Idiom         |
| Idiomatic Expression Interpretation    | IEI   | ROUGE-L, BERTScore-F1, METEOR, BLEU             | Idiom         |
| Noun Compound Compositionality         | NCC   | MCQ Accuracy                                    | Noun Compound |
| Noun Compound Extraction               | NCE   | Exact Match                                     | Noun Compound |
| Noun Compound Interpretation           | NCI   | ROUGE-L, BERTScore-F1, METEOR, BLEU             | Noun Compound |
| Lexical Collocation Categorization     | LCC   | Accuracy, Macro/Micro/Weighted F1               | Collocation   |
| Lexical Collocation Extraction         | LCE   | Exact Match                                     | Collocation   |
| Lexical Collocation Interpretation     | LCI   | ROUGE-L, BERTScore-F1, METEOR, BLEU             | Collocation   |
| Collocate Retrieval                    | CR    | Exact Match                                     | Collocation   |
| Collocation Identification             | CI    | Accuracy                                        | Collocation   |
| Verbal Multiword Expression Extraction | VMWE  | Exact Match                                     | Verbal MWE    |

Sequential (multi-step) tasks combine extraction with judgment or interpretation:
- Idiom / Collocation / Noun Compound Extraction + Judgment
- Idiom / Collocation / Noun Compound Extraction + Interpretation

## Supported Models

| Provider   | Models                                                                                      |
|------------|---------------------------------------------------------------------------------------------|
| OpenAI     | GPT-4 (0314/0613/Turbo/4o), GPT-5, o3   |
| Anthropic  | Claude 3 (Sonnet/Opus), Claude Sonnet 4.5              |
| Google     | Gemini 3.1 Pro, Gemma 3 27B IT                                    |
| DeepSeek   | DeepSeek-Chat, DeepSeek-R1                                                                  |
| Zhipu AI   | GLM-4.6                                                                                     |
| Alibaba    | Qwen3 (8B/14B/32B/235B-A22B)                                                               |
| Moonshot   | Kimi K2 Instruct                                                                            |
| Others       | Llama 3 (8B), Mistral 7B, Mixtral 8x7B  |

## Project Structure

```
SemanticQA/
├── resources/              # External datasets and source repositories
│   ├── dataset.zip         # Prepared benchmark data (unzip to use)
│   ├── AStitchInLanguageModels/  # Idiom & noun compound datasets
│   ├── ID10M/              # Idiom extraction dataset
│   ├── CollFrEn/           # French-English collocation data
│   ├── lexcomp/            # Lexical compositionality classifiers
│   ├── lexfunc/            # Lexical function data
│   ├── lexicalcollocations/# Collocation datasets
│   ├── LexNET/             # Lexical network corpus
│   ├── noun-compound-interpretation/  # NC interpretation data
│   ├── pronci/             # NC interpretation with transformers
│   └── graph-aware-collocation-recognition/
├── scripts/                # Data preparation & utility scripts
│   ├── data.py             # Data preparation for all tasks
│   ├── download_dataset.sh # Download raw datasets from sources
│   ├── calc_mean_sd.py     # Compute mean & std for BERT results
│   └── tsv2xlsx.py         # Convert TSV results to XLSX
├── semantic_qa/            # Main source code
│   ├── main.py             # Entry point for running evaluations
│   ├── args.py             # CLI argument definitions
│   ├── eval.py             # Evaluation metrics implementation
│   ├── utils.py            # I/O and prompt utilities
│   ├── data_utils.py       # Data loading & preprocessing
│   ├── model/              # Model query interfaces (OpenAI, Claude, Gemini, local)
│   ├── prompts/            # Zero-shot & few-shot prompt templates
│   ├── taxonomy/           # Semantic relation taxonomies (8/16 categories)
│   ├── training/           # Fine-tuning scripts (encoder LCC, T5 paraphrasing)
│   ├── type/               # Lexical function category mappings
│   ├── tests/              # Test scripts
│   └── results/            # Output directory
└── environment.yml         # Conda environment config
```

## Getting Started

### 1. Preparing Data

Download raw datasets:
```bash
# Available: asilm, id10m, pie, ncc, nci, nce, vmwe
./scripts/download_dataset.sh asilm
./scripts/download_dataset.sh pie
./scripts/download_dataset.sh vmwe
```

Or unzip the prepared benchmark data:
```bash
unzip resources/dataset.zip -d SemanticQA/
```

### 2. Setting Up the Environment
```bash
conda env create -f environment.yml
conda activate lexbench

cd semantic_qa
pip install -r requirements.txt
```

### 3. Running Evaluation

All evaluations are launched from `semantic_qa/` via `main.py`.

Example — idiom interpretation with `gpt-5` (zero-shot):
```bash
python main.py \
  --task idiom-paraphrase \
  --api_key <YOUR_API_KEY> \
  --model gpt-5 \
  --prompt_path prompts/idiom_paraphrase_zeroshot.txt \
  --example_path dataset/idiom_paraphrase/prepared/examples.tsv \
  --input_path dataset/idiom_paraphrase/prepared/idiom_paraphrase_prepared.tsv \
  --output_path results/idiom-paraphrase_0-shot_gpt-5.json \
  --evaluate \
  --shot_num 0 \
  --max_query 1000 \
  --max_tokens 128 \
  --temperature 0 \
  --presence_penalty 0 \
  --frequency_penalty 0
```

Key arguments:
| Argument              | Description                                      |
|-----------------------|--------------------------------------------------|
| `--task`              | Task name (see supported tasks above)            |
| `--model`             | Model identifier                                 |
| `--api_key`           | API key for the model provider                   |
| `--base_url`          | Custom API base URL (optional)                   |
| `--prompt_path`       | Path to prompt template file                     |
| `--example_path`      | Path to few-shot examples (optional)             |
| `--taxonomy_path`     | Path to taxonomy file (for LCC tasks)            |
| `--shot_num`          | Number of few-shot examples                      |
| `--evaluate`          | Run evaluation after generation                  |
| `--oracle_prompt`     | Use oracle prompt with ground truth hints         |
| `--max_query`         | Maximum number of API requests                   |
| `--max_tokens`        | Max generated tokens per request                 |
| `--temperature`       | Sampling temperature (0–2)                       |

### 4. Scaling-category Experiments (LCC)

Run collocation categorization across different taxonomy sizes (1/2/4/8/16 categories):
```bash
cd semantic_qa
./run_lcc_scaling.sh
```

### 5. Standalone Evaluation

Evaluate existing result files directly:
```bash
python eval.py --task lcc --result_file_path results/lcc_result.json
python eval.py --task vmwe --result_file_path results/vmwe_result.json
python eval.py --task iep --result_file_path results/idiom_paraphrase_result.json
```

### 6. Fine-tuning

Fine-tuning scripts are available under `semantic_qa/training/`:
- `collocation-categorization/` — Fine-tune encoder models for LCC
- `semantic-phrases-interpretation/` — Fine-tune T5 for paraphrasing tasks

## Citation

```bibtex
@article{liu2026revisiting,
    title={Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models},
    author={Liu, Yang and Li, Hongming and Qin, Melissa Xiaohui and Liu, Qiankun and Huang, Chao},
    journal={arXiv preprint arXiv:2604.16593},
    year={2026}
}
```

```bibtex
@article{liu2024revisiting,
    title={Revisiting a Pain in the Neck: Semantic Phrase Processing Benchmark for Language Models},
    author={Liu, Yang and Qin, Melissa Xiaohui and Li, Hongming and Huang, Chao},
    journal={arXiv preprint arXiv:2405.02861},
    year={2024}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
