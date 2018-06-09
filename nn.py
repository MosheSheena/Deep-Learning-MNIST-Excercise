# Imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from helper import timeit, clean_dir

# General vars for modularity
image_pixels = 28 ** 2
num_classes = 10
logistic_tensorboard_log_path = \
    "/Users/moshesheena/git/Deep-Learning-MNIST-Excercise/logs/logistic/"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def inference(
        x, w_initializer=tf.random_uniform_initializer,
        bias_initializer=tf.ones_initializer
):
    """
    produces a probabilty distribution over the output classes
    given a minibatch
    :param x: (matrix) the minibatch
    :param w_initializer:
    :param bias_initializer:
    :return: softmax output
    """
    tf.constant_initializer(value=0)
    W = tf.get_variable(
        "W", shape=[image_pixels, num_classes],
        initializer=w_initializer
    )
    b = tf.get_variable(
        "b", shape=[num_classes], initializer=bias_initializer
    )
    output = tf.nn.softmax(tf.matmul(x, W) + b)
    return output


def loss(output, y):
    """
    compute the average error per data sample using the Cross-Entropy function
    :param output: the inference of x
    :param y: data labels
    :return: loss over data
    """
    dot_product = y * tf.log(output)
    xentropy = -tf.reduce_sum(dot_product, reduction_indices=1)
    loss_val = tf.reduce_mean(xentropy)

    return loss_val


@timeit
def training(loss, global_step, learning_rate=0.5, logging=True):
    """
    train the model by computing the gradients and modifying the model params
    :param loss: (tensor) the value given from the loss function
    :param global_step:
    :return: accuracy
    """
    if logging:
        tf.summary.scalar("loss", loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluate(output, y):
    """
    evaluating the models accuracy
    :param output:
    :param y:
    :return:
    """
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("accuracy", accuracy)

    return accuracy


@timeit
def logistic(
        log_files_path, training_epochs, total_batch,
        learning_rate, batch_size=50, display_step=1
):

    clean_dir(logistic_tensorboard_log_path)

    with tf.Graph().as_default():

        x = tf.placeholder("float", [None, image_pixels])

        y = tf.placeholder("float", [None, num_classes])

        output = inference(x)
        cost = loss(output=output, y=y)
        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_op = training(cost, global_step=global_step,
                            learning_rate=learning_rate)
        eval_op = evaluate(output=output, y=y)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()

        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(
                log_files_path, graph_def=sess.graph_def
            )
        init_op = tf.initialize_all_variables()

        sess.run(init_op)

        # training cycle
        for epoch in range(training_epochs):
            avg_cost = 0

            # loop over all batches
            for i in range(total_batch):
                mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
                # Fit training using batch data
                feed_dict = {x: mbatch_x, y: mbatch_y}
                sess.run(train_op, feed_dict=feed_dict)
                # Compute average loss
                minibatch_cost = sess.run(cost, feed_dict=feed_dict)
                avg_cost += minibatch_cost/total_batch
                accuracy = sess.run(eval_op, feed_dict=feed_dict)

                print("Test{} Error:{}".format(i, accuracy))

        # Display logs per epoch step
        # if epoch % display_step == 0:
        val_feed_dict = {
            x: mnist.validation.images,
            y: mnist.validation.labels
        }
        accuracy = sess.run(eval_op, feed_dict=val_feed_dict)

        print("Validation Error:", 1-accuracy)

        summary_str = sess.run(summary_op, feed_dict=feed_dict)

        summary_writer.add_summary(
            summary_str, sess.run(global_step)
        )
        saver.save(sess, log_files_path, global_step=global_step)

        print("Optimization Finished!")

        test_feed_dict = {
            x: mnist.test.images,
            y: mnist.test.labels
        }

        accuracy = sess.run(eval_op, feed_dict=test_feed_dict)

        print("Test Accuracy:", accuracy)
