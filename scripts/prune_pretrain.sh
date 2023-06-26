#!/bin/bash

DEVICE=$1
PORT=$2
CKPT=$3
CUDA_VISIBLE_DEVICES=$DEVICE python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT prune.py --ckpt $CKPT --manner replace
