#!/bin/bash

# 不要使用劣质的合成数据!!!

model="ff"
epochs=2000
save_epochs=100
batch_size=1
learning_rate=1e-3
# ffn_layers=8,50,50,100,50,50,1
# ffn_layers=8,32,32,32,16,24,32,1
ffn_layers=8,16,16,1

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Training FFN with model: $model, epochs: $epochs, batch_size: $batch_size, learning_rate: $learning_rate" >> logs/train_ffn.log

checkpoint_dir="./outputs/ckpt/$model_${epochs}_${batch_size}_${learning_rate}"
mkdir -p $checkpoint_dir

python train.py --model "ff" \
    --data_paths "./data/Data_CHF_Zhao_2020_ATE.csv" \
    --checkpoint_dir $checkpoint_dir \
    --epochs $epochs --save_epochs $save_epochs --batch_size $batch_size \
    --learning_rate $learning_rate \
    --num_workers 4 \
    --seed 42
