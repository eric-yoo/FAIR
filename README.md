# FAIR

fairness and interpretability re-balancing

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
- [ ] experiment label_bias -> tracin with biased mnist
- [ ] integrate label_bias and tracin (naive, violation_tracin)
