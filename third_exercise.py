"""
Authors:
1. Moshe Sheena:    <ID>
2. Itay Ta'asiri:   <ID>
"""

# Imports
from sklearn.metrics import confusion_matrix
from math import ceil, sqrt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime
from timeit import default_timer as timer
import os
import shutil

# General vars for modularity
image_shape = 28 ** 2
num_classes = 10
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Helper functions
def clean_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)


def plot_image(image, img_shape, file_name):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')
    plt.savefig('{}.png'.format(file_name))


# Architectures
def logistic_regression(x, y, hyper_params):
    W = tf.get_variable(
        "W", shape=[image_shape, num_classes],
        initializer=hyper_params['w_initializer']
    )
    b = tf.get_variable(
        "b", shape=[num_classes], initializer=hyper_params['bias_initializer']
    )
    with tf.name_scope('logistic_layer'):
        z = tf.matmul(x, W) + b

    x_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
    )

    training_step = tf.train.AdamOptimizer(
        learning_rate=hyper_params['learning_rate']
    ).minimize(x_entropy)

    prediction = tf.argmax(input=z, dimension=1)
    true_label = tf.argmax(input=y, dimension=1)
    correct_pred = tf.equal(x=prediction, y=true_label)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return training_step, prediction, accuracy


def feed_forward_2_hidden_relu(x, y, hyper_params):
    num_hidden_neurons = hyper_params['num_hidden']

    W1 = tf.get_variable(
        "W1", shape=[image_shape, num_hidden_neurons],
        initializer=hyper_params['w_initializer']
    )
    b1 = tf.get_variable(
        "b1", shape=[num_hidden_neurons],
        initializer=hyper_params['bias_initializer']
    )
    W2 = tf.get_variable(
        "W2", shape=[num_hidden_neurons, num_hidden_neurons],
        initializer=hyper_params['w_initializer']
    )
    b2 = tf.get_variable(
        "b2", shape=[num_hidden_neurons],
        initializer=hyper_params['bias_initializer']
    )
    W3 = tf.get_variable(
        "W3", shape=[num_hidden_neurons, num_classes],
        initializer=hyper_params['w_initializer']
    )
    b3 = tf.get_variable(
        "b3", shape=[num_classes], initializer=hyper_params['bias_initializer']
    )

    z1 = tf.matmul(x, W1) + b1
    layer_1_logit = tf.nn.relu(z1)
    z2 = tf.matmul(layer_1_logit, W2) + b2
    layer_2_logit = tf.nn.relu(z2)
    z3 = tf.matmul(layer_2_logit, W3) + b3
    layer_3_logit = tf.nn.relu(z3)

    x_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z3)
    )

    training_step = tf.train.AdamOptimizer(
        learning_rate=hyper_params['learning_rate']
    ).minimize(x_entropy)

    prediction = tf.argmax(input=z3, dimension=1)
    true_label = tf.argmax(input=y, dimension=1)
    correct_pred = tf.equal(x=prediction, y=true_label)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return training_step, prediction, accuracy


def convolutional(x, y, hyper_params):
    pass


# Utilities
def run_architecture(
        x, y, hyper_params, training_step, prediction, accuracy, sess
):
    sess.run(tf.global_variables_initializer())

    for _ in range(hyper_params['num_batches']):
        minibatch_x, minibatch_y = mnist.train.next_batch(
            hyper_params['minibatch_size']
        )
        feed_dict = {x: minibatch_x, y: minibatch_y}

        sess.run(training_step, feed_dict)


def evaluate_architecture(x, y, accuracy, sess, against="test"):
    if against == "validation":
        feed_dict = {x: mnist.validation.images, y: mnist.validation.labels}
    elif against == "test":
        feed_dict = {x: mnist.test.images, y: mnist.test. labels}

    return sess.run(accuracy, feed_dict)


def plot_confusion_matrix(cls_pred, cls_true, architecture_name, file_name):
    """
    plot the confusion matrix and write the metrics:
    1. precision
    2. recall
    3. F_measure
    to the results file
    :param cls_pred:
    :param cls_true:
    :param architecture_name:
    :param file_name:
    :return:
    """
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    TN = []
    for i in range(num_classes):
        temp = np.delete(cm, i, 0)
        temp = np.delete(temp, i, 1)
        TN.append(sum(sum(temp)))

    precision = np.average(TP / (TP + FP))
    recall = np.average(TP / (TP + FN))
    f_measure = 2 * (precision * recall) / (precision + recall)

    with open('metrics_{}.txt'.format(architecture_name), 'a+') as res:
        res.write('confusion matrix:\n{}\n'.format(cm))
        res.write('precision: {}\n'.format(precision))
        res.write('recall: {}\n'.format(recall))
        res.write('f_measure: {}\n'.format(f_measure))

        # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # plt.show()
    plt.savefig('{}.png'.format(file_name))


def plot_conv_layer(layer, image, sess, x, file_name):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = sess.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = ceil(sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

            # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig('{}.png'.format(file_name))


def architecture_2_model(
        architecture, architecture_name, hyper_params, res_file_path):
    with open(res_file_path, 'a+') as res:
        now = datetime.datetime.now()
        res.write(
            "Results file {}-{}-{} {}:{}:{}\n".format(
                now.day, now.month, now.year, now.hour, now.minute, now.second
            )
        )

    # TODO:  <plot 1st ID figure>
    # digit_6 = mnist.test.images[11]
	# plot_image(digit_6, 'my_digit_id')

    with open(res_file_path, 'a+') as res:
        res.write("Architecture: {}\n".format(architecture_name))

    sess = tf.Session()

    x = tf.placeholder(tf.float32, [None, image_shape])
    y = tf.placeholder(tf.float32, [None, num_classes])

    # Measure CPU time
    s_time = timer()

    training_step, prediction, accuracy = architecture(x, y, hyper_params)
    run_architecture(
        x, y, hyper_params, training_step,
        prediction, accuracy, sess
    )
    validation_accuracy = evaluate_architecture(
        x, y, accuracy, sess, against="validation")
    test_accuracy = evaluate_architecture(
        x, y, accuracy, sess, against="test"
    )

    # Measure CPU time
    e_time = timer()

    cls_true = tf.argmax(mnist.test.labels, axis=1)
    c1, c2 = sess.run(
        [cls_true, prediction], feed_dict={
            x: mnist.test.images, y: mnist.test.labels
        }
    )
    plot_confusion_matrix(
        c1, c2, architecture_name, 'confusion_mat_{}'.format(architecture_name)
    )

    num_weights = np.sum(
        [
            np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()
        ]
    )

    with open(res_file_path, 'a+') as res:
        res.write('runtime(CPU): {}\n'.format(e_time-s_time))
        res.write('num weights: {}\n'.format(num_weights))
        res.write('training accuracy: {}\n'.format(accuracy))
        res.write('validation accuracy: {}\n'.format(validation_accuracy))
        res.write('test accuracy: {}\n'.format(test_accuracy))


def execute_excercise():
    hyper_params = {
        'w_initializer': tf.random_uniform_initializer,
        'bias_initializer': tf.ones_initializer,
        'learning_rate': 0.01,
        'num_batches': 1200,
        'minibatch_size': 50
    }
    architecture_2_model(
        architecture=logistic_regression,
        architecture_name='Logistic Regression',
        hyper_params=hyper_params,
        res_file_path="nn_res_file.txt"
    )

    hyper_params = {
        'w_initializer': tf.random_normal_initializer,
        'bias_initializer': tf.constant_initializer(value=0),
        'learning_rate': 0.01,
        'num_batches': 1200,
        'minibatch_size': 50,
        'num_hidden': 200
    }
    architecture_2_model(
        architecture=feed_forward_2_hidden_relu,
        architecture_name='Feed Forward 2 relu hidden layers',
        hyper_params=hyper_params,
        res_file_path="nn_res_file.txt"
    )


if __name__ == '__main__':
    execute_excercise()
