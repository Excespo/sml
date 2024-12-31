#!/bin/bash

model="liu"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running with model: $model" >> logs/train_liu.log

python train.py --model "liu" --batch_size 1 \
    --data_paths "./data/Data_CHF_Zhao_2020_ATE.csv" \
    --num_workers 4 \
    --seed 42
