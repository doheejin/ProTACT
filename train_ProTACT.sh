#!/usr/bin/env bash
model_name='ProTACT'
for seed in 12 22 32 42 52
do
    for prompt in {1..8}
    do
        python train_ProTACT.py --test_prompt_id ${prompt} --model_name ${model_name} --seed ${seed} --num_heads 2 --features_path 'data/LDA/hand_crafted_final_'
    done
done