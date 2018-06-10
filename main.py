# Imports
import nn as mnist_ex

# Vars
hyperparams_logistic = {
    "training_epochs": 100,
    "total_batch": 1200,
    "batch_size": 50,
    "learning_rate": 0.01
}


def execution():
    mnist_ex.logistic(**hyperparams_logistic)

