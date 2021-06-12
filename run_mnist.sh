# CUDA_VISIBLE_DEVICES=1 python3 run_label_bias.py
#CUDA_VISIBLE_DEVICES=1 python3 run_tracin.py
mode=$1

if [ $mode == "baseline" ]
then
    echo "Running baseline"
    CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --dataset mnist
elif [ $mode == "ideal" ]
then
    echo "Running ideal"
    CUDA_VISIBLE_DEVICES=1 python3 run_ideal.py --dataset mnist
elif [ $mode == "lb" ]
then
    echo "Running Label Bias"
    CUDA_VISIBLE_DEVICES=1 python3 run_label_bias.py --dataset mnist
elif [ $mode == "fair_naive" ]
then
    echo "Running FAIR Naive"
    CUDA_VISIBLE_DEVICES=1 python3 run_FAIR_naive.py --dataset mnist 
elif [ $mode == "fair" ]
then
    echo "Running FAIR"
    CUDA_VISIBLE_DEVICES=1 python3 run_FAIR.py --dataset mnist
else 
    echo "No such mode : terminating"
fi