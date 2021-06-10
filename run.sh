# CUDA_VISIBLE_DEVICES=1 python3 run_label_bias.py
#CUDA_VISIBLE_DEVICES=1 python3 run_tracin.py
mode=$1
if [ $mode == "fair" ]
then
    echo "Running FAIR"
    CUDA_VISIBLE_DEVICES=1 python3 run_FAIR.py --dataset mnist --poisoned_ratio 0.3
elif [ $mode == "rcl" ]    
then
    echo "Running RCL"
    CUDA_VISIBLE_DEVICES=1 python3 run_RCL_single.py --dataset mnist --poisoned_ratio 0.3
else 
    echo "No such mode : terminating"
fi
