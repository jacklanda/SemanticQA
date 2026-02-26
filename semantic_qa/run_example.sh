#!/bin/bash

python main.py \
  --task noun-compound-compositionality \
  --api_key "<Your API KEY>" \
  --model claude-instant-1 \
  --prompt_path prompts/noun_compound_compositionality_fewshot.txt \
  --example_path dataset/noun_compound_compositionality/prepared/examples.tsv \
  --input_path dataset/noun_compound_compositionality/prepared/noun_compound_compositionality_prepared.tsv \
  --output_path results/noun-compound-compositionality_5-shot_claude-instant-1.json \
  --evaluate \
  --shot_num 5 \
  --max_query 1000 \
  --max_tokens 128 \
  --temperature 0 \
  --presence_penalty 0 \
  --frequency_penalty 0
