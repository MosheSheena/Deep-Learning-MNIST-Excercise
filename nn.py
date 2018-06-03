# Imports
import tensorflow as tf

# General vars for modularity
image_pixels = 28 ** 2
num_classes = 10


def inference(x):
    """
    produces a probabilty distribution over the output classes
    given a minibatch
    :param x: (matrix) the minibatch
    :return:  (tf.nn.softmax) returns softmax output
    """
    tf.constant_initializer(value=0)
    W = tf.get_variable(
        "W", shape=[image_pixels, num_classes],
        initializer=tf.random_uniform_initializer
    )
    b = tf.get_variable(
        "b", shape=[num_classes], initializer=tf.ones_initializer
    )
    output = tf.nn.softmax(tf.matmul(x, W) + b)
    return output


def loss(output, y):
    """
    compute the average error per data sample
    :param output: the inference of x
    :param y: data labels
    :return: loss over data
    """
    dot_product = y * tf.log(output)
    xentropy = -tf.reduce_sum(dot_product, reduction_indices=1)
    loss_val = tf.reduce_mean(xentropy)

    return loss_val


def training(cost, global_step, learning_rate=None):
    """
    train the model by computing the gradients and modifying the model params
    :param cost: (tensor) the value given from the loss function
    :param global_step:
    :return: accuracy
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)

    return train_op


def evaluate(output, y):
    """

    :param output:
    :param y:
    :return:
    """
    correct_prediction = tf.equal(
        tf.arg_max(output, 1), tf.argmax(y, 1)
    )
    accuracy = tf.reduce_mean(
        tf.cast(correct_prediction, tf.float32)
    )

    return accuracy


