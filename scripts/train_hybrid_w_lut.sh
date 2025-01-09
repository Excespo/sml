#!/bin/bash

model="hybrid-w-lut"
epochs=1000
save_epochs=100
batch_size=1
learning_rate=1e-3
ffn_layers=6,16,16,1

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Training Hybrid-w-lut with model: $model, epochs: $epochs, batch_size: $batch_size, learning_rate: $learning_rate" >> logs/train_hybrid_w_lut.log

checkpoint_dir="./outputs/ckpts/${model}_${epochs}_${batch_size}_${learning_rate}_[${ffn_layers}]"
mkdir -p $checkpoint_dir

python train.py --model $model --ffn_layers $ffn_layers \
    --data_paths "./data/Data_CHF_Zhao_2020_ATE.csv" \
    --checkpoint_dir $checkpoint_dir \
    --epochs $epochs --save_epochs $save_epochs --batch_size $batch_size \
    --learning_rate $learning_rate \
    --num_workers 4 \
    --seed 42
