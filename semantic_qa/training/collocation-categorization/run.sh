#!/bin/zsh

export CUDA_VISIBLE_DEVICES=0,1

for order in "1-0" "1-1" "1-2" "2-0" "2-1" "2-2" "4-0" "4-1" "4-2" "8-0" "8-1" "8-2" "16-0" "16-1" "16-2"
do
    #echo "order: $order | model: bert-base-uncased | do_conditioning: False | eval_on_test: True"
    #nohup python finetune_encoders_lcc.py \
        #--model bert-base-uncased \
        #--eval_on_test > "logs/bert-base-uncased-$order.log" 2>&1 &

    echo "order: $order | model: bert-base-uncased | do_conditioning: True | eval_on_test: True"
    nohup python finetune_encoders_lcc.py \
        --model bert-base-uncased \
        --do_conditioning \
        --eval_on_test > "logs/bert-base-uncased-$order-conditioning.log" 2>&1

    #echo "order: $order | model: bert-large-uncased | do_conditioning: False | eval_on_test: True"
    #nohup python finetune_encoders_lcc.py \
        #--model bert-large-uncased \
        #--eval_on_test > "logs/bert-large-uncased-$order.log" 2>&1 &

    echo "order: $order | model: bert-large-uncased | do_conditioning: True | eval_on_test: True"
    nohup python finetune_encoders_lcc.py \
        --model bert-large-uncased \
        --do_conditioning \
        --eval_on_test > "logs/bert-large-uncased-$order-conditioning.log" 2>&1
done
