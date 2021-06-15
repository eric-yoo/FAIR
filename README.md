# FAIR

fairness and interpretability re-balancing

## datasets

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## env

developed with Python 3.7.6

## install
  
```Python
pip install -r requirements
```

## run

- label_bias

    ```Python
    python run_label_bias.py
    ```

- tracin

    ```Python
    python run_tracin.py
    ```

## todo

- [x] use mnist dataset (DatasetBuilder)
- [x] add id column in dataset (need to check)
- [x] implement mlp instead of resnet in tracin (see tracin branch)
- [x] refactor tracin (see main branch)
- [X] fix tracin model issue (labels don't match actual numbers)
- [X] use tf2 in label_bias, check compatibility with tracin
- [X] add preprocessing in tracin data
- [X] implement tracin self-influence
- [X] experiment tracin with biased mnist
- [X] fix label_bias bug (incorrect eval accuracy)
- [X] fix label_bias bug (rebalance weight not applied)
- [X] experiment label_bias -> tracin with biased mnist
- [ ] integrate label_bias and tracin (naive, violation_tracin)

## jot

- reverse label poisoning (RLP)을 섞는다
- 학습 순서 adversarially 주기 (FL 가정)
- debias_weights_TI (지금 한거 어떻게든 살려보기)
- 적당히 잘 학습된 모델 (30%) -> good/poisoned/RLP/noise/mislabel 섞인 data 들어올때 (10%) (CL 가정)
- 적당히 잘 학습된 모델 (30%) -> poisoned 섞인 data 들어올때 (70%) (현실적인 딥러닝 서비스 가정)

many to 2 polution

2 to many polution
train[20%:] [5005, 5480, 2421, 5153, 4952, 4880, 4968, 5131, 4955, 5055]
train[20%:] [4705, 5433, 4772, 4936, 4681, 4333, 4728, 4966, 4703, 4743]