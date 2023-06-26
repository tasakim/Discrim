#!/bin/bash

DEVICE=$1
NUM_DEVICES=$2
PORT=$3
CKPT=$4
CUDA_VISIBLE_DEVICES=$DEVICE python -m torch.distributed.launch --nproc_per_node=$NUM_DEVICES --master_port=$PORT eagle_prune.py --ckpt $CKPT --manner replace
