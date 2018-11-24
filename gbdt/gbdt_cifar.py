import numpy as np
from collections import namedtuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

NAVY = "#001f3f"


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


Data = namedtuple("Data", ['X', 'y'])


def load_training(path, num_batches=1):
    batch_one = unpickle(path + "data_batch_1")
    training_features = batch_one[b'data']
    training_labels = np.array(batch_one[b'labels'])

    for idx in range(1, num_batches):
        batch = unpickle(path + "data_batch_{}".format(str(idx + 1)))
        training_features = np.append(training_features, batch[b'data'], axis=0)
        training_labels = np.append(training_labels, np.array(batch[b'labels']))
    return Data(X=training_features, y=training_labels)


def load_test(path):
    test_batch = unpickle(path + "test_batch")
    test_features = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])
    return Data(X=test_features, y=test_labels)


def pre_process(train, test, scaler=StandardScaler()):
    """

    :param train: Data namedtuple
    :param test: Data namedtuple
    :param scaler: Scaling mechanism to be used
    :return: Scaled versions of both data sets
    """
    scaler.fit(train.X)
    return Data(X=scaler.transform(train.X), y=train.y), Data(X=scaler.transform(test.X), y=test.y)


def experiment(estimator, preprocessed_train, hyper_parameter_name, hyper_parameter_values, x_axis_label):
    train_scores, valid_scores = validation_curve(estimator,
                                                  preprocessed_train.X,
                                                  preprocessed_train.y,
                                                  hyper_parameter_name,
                                                  hyper_parameter_values,
                                                  cv=3,
                                                  verbose=3,
                                                  n_jobs=3,
                                                  scoring="accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)
    plt.title("Validation Curve with GBDT")
    plt.xlabel(x_axis_label)
    plt.ylabel("Accuracy")
    plt.xticks(hyper_parameter_values)
    lw = 2

    plt.plot(hyper_parameter_values, train_scores_mean, label="Training Accuracy",
                 color="darkorange", lw=lw)
    plt.fill_between(hyper_parameter_values, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(hyper_parameter_values, valid_scores_mean, label="Cross-Validation Accuracy",
                 color=NAVY, lw=lw)
    plt.fill_between(hyper_parameter_values, valid_scores_mean - valid_scores_std,
                     valid_scores_mean + valid_scores_std, alpha=0.2,
                     color=NAVY, lw=lw)
    print("Training Accuracy:", train_scores)
    print("Valid Accuracy:", valid_scores)
    plt.legend(loc="best")
    plt.savefig(hyper_parameter_name + "_cv.png", format="png")


path = '/Users/adityajoshi/Documents/IFT6390/fundamentals-of-ml-project-fall-2018/data/cifar_10/cifar-10-batches-py/'

training_data = load_training(path, num_batches=2)
test_data = load_test(path)

preprocessed_training_data, preprocessed_test_data = pre_process(training_data, test_data)

tree_depths = [3, 5, 10]
learning_rates = [0.001, 0.1, 0.5]  # regularization

# Experiment 1: CV on Number of Boosting Stages.
experiment(GradientBoostingClassifier(max_depth=3),
           preprocessed_training_data,
           "n_estimators",
           [10, 100, 500, 1000],
           "Number of Estimators")



