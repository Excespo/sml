#!/bin/bash

model="lut"
batch_size=1

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Training lut with model: $model, batch_size: $batch_size" >> logs/train_lut.log

checkpoint_dir="./outputs/ckpts/${model}_${batch_size}"
mkdir -p $checkpoint_dir

python train.py --model $model \
    --data_paths "./data/Data_CHF_Zhao_2020_ATE.csv" \
    --checkpoint_dir $checkpoint_dir \
    --batch_size $batch_size \
    --num_workers 4 \
    --seed 42
