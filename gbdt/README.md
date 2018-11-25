# CIFAR-10 (Only 10k images were used for training)


## Cross-Validation 
* Number of Estimators
![alt text](https://github.com/adijo/fundamentals-of-ml-project-fall-2018/blob/master/gbdt/figures/cifar_n_estimators_cv.png)

* Max Depth
![alt text](https://github.com/adijo/fundamentals-of-ml-project-fall-2018/blob/master/gbdt/figures/cifar_max_depth_cv.png)

* Learning Rate
![alt text](https://github.com/adijo/fundamentals-of-ml-project-fall-2018/blob/master/gbdt/figures/cifar_learning_rate_cv.png)

## Generalization: Trained on batch 1 (10k images)
Estimators: 160, Max Depth: 5, Learning Rate: 0.1
* Generalization Confusion Matrix
![alt text](https://github.com/adijo/fundamentals-of-ml-project-fall-2018/blob/master/gbdt/figures/cifar_generalization_confusion_matrix.png)

* Accuracy: 0.4783
* Classification Report

| `precision` |     `recall`  | `f1-score`  |  `support`|
| --- | --- | --- | --- | --- |
0  |     0.56  |    0.57 |      0.57 |      1000 |
1   |    0.57  |    0.51  |    0.54 |     1000|
2   |    0.34  |    0.37   |   0.36 |     1000|
3  |     0.33  |    0.34  |    0.33 |     1000|
4  |     0.41  |    0.40  |    0.40 |     1000|
5  |     0.41 |     0.38  |    0.39  |    1000|
6  |     0.51  |    0.57  |    0.54  |    1000|
7   |    0.54  |    0.47   |   0.51  |    1000|
8  |     0.59  |    0.63  |    0.61  |    1000|
9   |    0.53  |    0.54 |     0.53 |     1000|
avg / total     |  0.48   |   0.48   |   0.48   |  10000|

# Banking

## Cross-Validation 
* Number of Estimators
![alt text](https://github.com/adijo/fundamentals-of-ml-project-fall-2018/blob/master/gbdt/figures/banking_n_estimators_cv.png)

* Max Depth
![alt text](https://github.com/adijo/fundamentals-of-ml-project-fall-2018/blob/master/gbdt/figures/banking_max_depth_cv.png)

* Learning Rate
![alt text](https://github.com/adijo/fundamentals-of-ml-project-fall-2018/blob/master/gbdt/figures/banking_learning_rate_cv.png)

## Generalization: Trained on the full Dataset
Estimators: 160, Max Depth: 5, Learning Rate: 0.1


* Generalization Confusion Matrix
![alt text](https://github.com/adijo/fundamentals-of-ml-project-fall-2018/blob/master/gbdt/figures/banking_generalization_confusion_matrix.png)


