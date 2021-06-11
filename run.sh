# CUDA_VISIBLE_DEVICES=1 python3 run_label_bias.py
#CUDA_VISIBLE_DEVICES=1 python3 run_tracin.py
mode=$1
poisoned_ratio=$2

if [ -v $poisoned_ratio ]
then
    poisoned_ratio=0.3
fi

if [ $mode == "baseline" ]
then
    echo "Running baseline"
    CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --dataset mnist --poisoned_ratio $poisoned_ratio
elif [ $mode == "ideal" ]
then
    echo "Running ideal"
    CUDA_VISIBLE_DEVICES=1 python3 run_ideal.py --dataset mnist --poisoned_ratio $poisoned_ratio
elif [ $mode == "label_bias" ]
then
    echo "Running Label Bias"
    CUDA_VISIBLE_DEVICES=1 python3 run_label_bias.py --dataset mnist --poisoned_ratio $poisoned_ratio
elif [ $mode == "fair_naive" ]
then
    echo "Running FAIR Naive"
    CUDA_VISIBLE_DEVICES=1 python3 run_FAIR_naive.py --dataset mnist --poisoned_ratio $poisoned_ratio
elif [ $mode == "fair" ]
then
    echo "Running FAIR"
    CUDA_VISIBLE_DEVICES=1 python3 run_FAIR.py --dataset mnist --poisoned_ratio $poisoned_ratio
else 
    echo "No such mode : terminating"
fi
