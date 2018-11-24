import pandas as pd
import numpy as np
from collections import namedtuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn_evaluation import plot
from sklearn.metrics import classification_report

NAVY = "#001f3f"


Data = namedtuple("Data", ['X', 'y'])


def experiment(estimator, preprocessed_train, hyper_parameter_name, hyper_parameter_values, x_axis_label):
    train_scores, valid_scores = validation_curve(estimator,
                                                  preprocessed_train.X,
                                                  preprocessed_train.y,
                                                  hyper_parameter_name,
                                                  hyper_parameter_values,
                                                  cv=4,
                                                  verbose=3,
                                                  n_jobs=3,
                                                  scoring="accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)
    plt.title("Validation Curve with GBDT for the Banking Dataset")
    plt.xlabel(x_axis_label)
    plt.ylabel("Accuracy")
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
    plt.savefig("banking_" + hyper_parameter_name + "_cv.png", format="png")


def evaluate_generalization(test_dataset, estimator):
    predictions = estimator.predict(test_dataset.X)
    true_labels = test_dataset.y
    accuracy = accuracy_score(true_labels, predictions)
    loss = log_loss(true_labels, predictions)
    ax = plot.confusion_matrix(true_labels, predictions)
    plt.subplot(ax)
    plt.savefig("generalization_confusion_matrix.png", format="png")

    print("Accuracy", accuracy)
    print("Log Loss", loss)
    print(classification_report(true_labels, predictions))


training_data = pd.read_csv('../data/banking/bank-additional-full-transformed-train.csv', sep=';')
training_data = Data(X=training_data.iloc[:, :-1], y=training_data.iloc[:, -1])

Data = namedtuple("Data", ['X', 'y'])

# # Experiment 1: CV on Number of Boosting Stages.
# experiment(GradientBoostingClassifier(max_depth=3, learning_rate=0.1),
#            training_data,
#            "n_estimators",
#            [10, 20, 50, 100, 150, 200, 300, 500, 1000],
#            "Number of Estimators")


# # Experiment 2: CV on Max Tree Depth
# experiment(GradientBoostingClassifier(n_estimators=160, learning_rate=0.1),
#            training_data,
#            "max_depth",
#            [1, 2, 5, 10, 15, 20],
#            "Max Tree Depth")

# # Experiment 3: CV on the Learning Rate (Regularization)
# experiment(GradientBoostingClassifier(n_estimators=160, max_depth=5),
#            training_data,
#            "learning_rate",
#            [0.0001, 0.001, 0.1, 0.5, 0.9, 1],
#            "Learning Rate")

estimator = GradientBoostingClassifier(n_estimators=160, learning_rate=0.1, max_depth=5)
estimator.fit(training_data.X, training_data.y)
test_data = pd.read_csv('../data/banking/bank-additional-full-transformed-test.csv', sep=';')
test_data = Data(X=test_data.iloc[:, :-1], y=test_data.iloc[:, -1])
evaluate_generalization(test_data, estimator)
