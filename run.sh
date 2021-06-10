#!/bin/bash

mode=$1
poisoned_ratio=$2
pretrain_ratio=$3

if [ -z $poisoned_ratio ]
then
    poisoned_ratio=0.4
fi

if [ -z $pretrain_ratio ]
then
    pretrain_ratio=5%
fi

echo "Poisoned Ratio: "${poisoned_ratio}
echo "Pretrain Ratio: "${pretrain_ratio}


if [ $mode == "lb" ]
then
    echo "Running Label Bias"
    CUDA_VISIBLE_DEVICES=1 python3 run_label_bias.py --dataset mnist --poisoned_ratio $poisoned_ratio --pretrain_ratio $pretrain_ratio
elif [ $mode == "fair" ]
then
    echo "Running FAIR"
    CUDA_VISIBLE_DEVICES=1 python3 run_FAIR.py --dataset mnist --poisoned_ratio $poisoned_ratio --pretrain_ratio $pretrain_ratio
elif [ $mode == "rcl" ]    
then
    echo "Running RCL"
    CUDA_VISIBLE_DEVICES=1 python3 run_RCL_single.py --dataset mnist --poisoned_ratio $poisoned_ratio --pretrain_ratio $pretrain_ratio
else 
    echo "No such mode : terminating"
fi