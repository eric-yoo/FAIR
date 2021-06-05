import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--poisoned_ratio', type=float, default=0.2)
parser.add_argument('--poisoned_label', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=5)
parser.add_argument('--pretrain_ratio', type=str, default='50%')

args = parser.parse_args()
print(args)

CHECKPOINTS_PATH_FORMAT = "simpleNN/checkpoints/{}_iter{}_ckpt{}"
FAIR_PATH_FORMAT = "simpleNN/checkpoints/fair_iter{}_ckpt{}"
TRACIN_PATH = "simpleNN/checkpoints/ckpt{}" 
PRETRAINED_PATH = "simpleNN/checkpoints/pretrain_{}_ckpt{}" 