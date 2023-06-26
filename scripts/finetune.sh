#!/bin/bash
DEVICE=$1
PORT=$2
NUM_DEVICES=$3

CUDA_VISIBLE_DEVICES=$DEVICE python -m torch.distributed.launch --nproc_per_node=$NUM_DEVICES --master_port=$PORT finetune.py --checkpoint pruned_model.pt
