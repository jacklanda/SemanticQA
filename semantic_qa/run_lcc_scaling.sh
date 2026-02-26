#!/bin/bash

API_KEY="<Your API Key>"
BASE_URL="https://api.openai.com/v1"

scaling_gpt35_1(){
    for model in "gpt-3.5-turbo-0613" 
    do
        for order in "1-0" "1-1" "1-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --base_url ${BASE_URL} \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_gpt35_2(){
    for model in "gpt-3.5-turbo-0613" 
    do
        for order in "2-0" "2-1" "2-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --base_url $BASE_URL \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_gpt35_2_fewshot(){
    for model in "gpt-3.5-turbo-0613" 
    do
        for order in "2-0" "2-1" "2-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --base_url $BASE_URL \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_gpt4_2(){
    for model in "gpt-4-1106-preview" 
    do
        for order in "2-0" "2-1" "2-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --base_url $BASE_URL \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_gpt4_2_fewshot(){
    for model in "gpt-4-1106-preview" 
    do
        for order in "2-0" "2-1" "2-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --base_url $BASE_URL \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_gpt35_4(){
    for model in "gpt-3.5-turbo-0613" 
    do
        for order in "4-0" "4-1" "4-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --base_url $BASE_URL \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_gpt35_4_fewshot(){
    for model in "gpt-3.5-turbo-0613" 
    do
        for order in "4-0" "4-1" "4-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --base_url $BASE_URL \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_gpt4_4(){
    for model in "gpt-4-1106-preview" 
    do
        for order in "4-0" "4-1" "4-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --base_url $BASE_URL \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_gpt4_4_fewshot(){
    for model in "gpt-4-1106-preview"
    do
        for order in "4-0" "4-1" "4-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --base_url $BASE_URL \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_gpt35_8(){
    for model in "gpt-3.5-turbo-0613" 
    do
        for order in "8-0" "8-1" "8-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --base_url $BASE_URL \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_gpt35_8_fewshot(){
    for model in "gpt-3.5-turbo-0613" 
    do
        for order in "8-0" "8-1" "8-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --base_url $BASE_URL \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_gpt4_8(){
    for model in "gpt-4-1106-preview" 
    do
        for order in "8-2"  # "8-0" "8-1" 
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --base_url $BASE_URL \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_gpt4_8_fewshot(){
    for model in "gpt-4-1106-preview"
    do
        for order in "8-0" "8-1" "8-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --base_url $BASE_URL \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_gpt35_16(){
    for model in "gpt-3.5-turbo-0613" 
    do
        for order in "16-0" "16-1" "16-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --base_url $BASE_URL \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_gpt35_16_fewshot(){
    for model in "gpt-3.5-turbo-0613" 
    do
        for order in "16-0" "16-1" "16-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --base_url $BASE_URL \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_gpt4_16(){
    for model in "gpt-4-1106-preview"
    do
        for order in "16-0" "16-1" "16-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --base_url $BASE_URL \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_gpt4_16_fewshot(){
    for model in "gpt-4-1106-preview"
    do
        for order in "16-0" "16-1" "16-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --base_url $BASE_URL \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_claude3_2(){
    for model in "claude-3-opus-20240229" 
    do
        for order in "2-0" "2-1" "2-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_claude3_2_fewshot(){
    for model in "claude-3-opus-20240229" 
    do
        for order in "2-0" "2-1" "2-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_claude3_4(){
    for model in "claude-3-opus-20240229" 
    do
        for order in "4-0" "4-1" "4-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_claude3_4_fewshot(){
    for model in "claude-3-opus-20240229" 
    do
        for order in "4-0" "4-1" "4-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_claude3_8(){
    for model in "claude-3-opus-20240229" 
    do
        for order in "8-0" "8-1" "8-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_claude3_8_fewshot(){
    for model in "claude-3-opus-20240229" 
    do
        for order in "8-0" "8-1" "8-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_claude3_16(){
    for model in "claude-3-opus-20240229" 
    do
        for order in "16-0" "16-1" "16-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_claude3_16_fewshot(){
    for model in "claude-3-opus-20240229" 
    do
        for order in "16-0" "16-1" "16-2"
        do
            for shot_num in 3
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_gemini_2(){
    for model in "gemini-pro" 
    do
        for order in "2-0" "2-1" "2-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_gemini_2_fewshot(){
    for model in "gemini-pro" 
    do
        for order in "2-0" "2-1" "2-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_gemini_4(){
    for model in "gemini-pro" 
    do
        for order in "4-0" "4-1" "4-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_gemini_4_fewshot(){
    for model in "gemini-pro" 
    do
        for order in "4-0" "4-1" "4-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_gemini_8(){
    for model in "gemini-pro" 
    do
        for order in "8-0" "8-1" "8-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_gemini_8_fewshot(){
    for model in "gemini-pro" 
    do
        for order in "8-0" "8-1" "8-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

scaling_gemini_16(){
    for model in "gemini-pro" 
    do
        for order in "16-0" "16-1" "16-2"
        do
            echo "Running scaling exp \"collocation-categorization\" with ($model) ($order) ...\n"
            python main.py \
            --task collocation-categorization \
            --api_key $API_KEY \
            --model $model \
            --prompt_path prompts/collocation_categorization_zeroshot.txt \
            --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy.tsv" \
            --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
            --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
            --output_path "results/collocation-categorization_0-shot_scaling-"$order"_"$model".json" \
            --evaluate \
            --shot_num 0 \
            --max_query 1000 \
            --max_tokens 128 \
            --temperature 0 \
            --presence_penalty 0 \
            --frequency_penalty 0
        done
    done
}

scaling_gemini_16_fewshot(){
    for model in "gemini-pro" 
    do
        for order in "16-0" "16-1" "16-2"
        do
            for shot_num in 3 5
            do
                echo "Running scaling exp \"collocation-categorization\" with ($model) ($shot_num-shot) ($order) ...\n"
                python main.py \
                --task collocation-categorization \
                --api_key $API_KEY \
                --model $model \
                --prompt_path prompts/collocation_categorization_fewshot.txt \
                --taxonomy_path "dataset/collocation_categorization/scaling/$order/taxonomy_$shot_num-shot.tsv" \
                --example_path "dataset/collocation_categorization/scaling/$order/example.tsv" \
                --input_path "dataset/collocation_categorization/scaling/$order/test.tsv" \
                --output_path "results/collocation-categorization_$shot_num-shot_scaling-"$order"_"$model".json" \
                --evaluate \
                --shot_num $shot_num \
                --max_query 1000 \
                --max_tokens 128 \
                --temperature 0 \
                --presence_penalty 0 \
                --frequency_penalty 0
            done
        done
    done
}

# OpenAI Models
scaling_gpt35_1
scaling_gpt35_2
scaling_gpt35_2_fewshot
scaling_gpt4_2
scaling_gpt4_2_fewshot
scaling_gpt35_4
scaling_gpt35_4_fewshot
scaling_gpt4_4
scaling_gpt4_4_fewshot
scaling_gpt35_4
scaling_gpt35_8
scaling_gpt35_8_fewshot
scaling_gpt4_8_fewshot
scaling_gpt4_8
scaling_gpt35_16
scaling_gpt35_16_fewshot
scaling_gpt4_16_fewshot
scaling_gpt4_16

# Anthropic Models
scaling_claude3_2
scaling_claude3_2_fewshot
scaling_claude3_4
scaling_claude3_4_fewshot
scaling_claude3_8
scaling_claude3_8_fewshot
scaling_claude3_16
scaling_claude3_16_fewshot

# Google Models
scaling_gemini_2
scaling_gemini_2_fewshot
scaling_gemini_4
scaling_gemini_4_fewshot
scaling_gemini_8
scaling_gemini_8_fewshot
scaling_gemini_16
scaling_gemini_16_fewshot
