# CUDA_VISIBLE_DEVICES=1 python3 run_label_bias.py
#CUDA_VISIBLE_DEVICES=1 python3 run_tracin.py
dataset=$1

echo "Running baseline"
CUDA_VISIBLE_DEVICES=1 python3 run_baseline.py --dataset $dataset --n_epochs 10
echo "Running ideal"
CUDA_VISIBLE_DEVICES=1 python3 run_ideal.py --dataset $dataset --n_epochs 10
echo "Running Label Bias"
CUDA_VISIBLE_DEVICES=1 python3 run_label_bias.py --dataset $dataset --n_epochs 10
echo "Running FAIR"
CUDA_VISIBLE_DEVICES=1 python3 run_FAIR.py --dataset $dataset --n_epochs 10