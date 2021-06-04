import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--corrupt_ratio', type=float, default=0.2)
parser.add_argument('--biased_label', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=120)

args = parser.parse_args()
print(args)

CHECKPOINTS_PATH_FORMAT = "simpleNN/checkpoints/lb_iter{}_ckpt{}"
FAIR_PATH_FORMAT = "simpleNN/checkpoints/fair_iter{}_ckpt{}"
TRACIN_PATH = "simpleNN/checkpoints/ckpt{}" 
