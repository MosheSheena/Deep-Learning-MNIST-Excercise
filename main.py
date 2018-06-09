# Imports
import nn as mnist_ex

# Vars
hyperparams_logistic = {
    "log_files_path": mnist_ex.logistic_tensorboard_log_path,
    "training_epochs": 100,
    "total_batch": 1200,
    "batch_size": 50,
    "learning_rate": 0.01
}

if __name__ == "__main__":
    mnist_ex.logistic(**hyperparams_logistic)
