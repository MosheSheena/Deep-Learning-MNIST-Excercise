# Imports
import tensorflow as tf

# General vars for modularity
image_pixels = 28 ** 2
num_classes = 10


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


def training(loss, global_step, learning_rate=None, logging=False):
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
    evaluating the model on the validation / test-set
    :param output:
    :param y:
    :return:
    """
    correct_prediction = tf.equal(tf.arg_max(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy


def logistic():

    with tf.Graph().as_default():

        x = tf.placeholder("float", [None, image_pixels])

        y = tf.placeholder("float", [None, num_classes])

        output = inference(x)
        cost = loss(output=output, y=y)
        global_step = tf.Variable(0, name='global step', trainable=False)

        train_op = training(cost, global_step=global_step)
        eval_op = evaluate(output=output, y=y)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()

        sess = tf.Session()
        summary_writer = tf.summary.FileWriter\
            (
                "/Users/moshesheena/git/Deep-Learning-MNIST-Excercise/logs",
                graph_def=sess.graph_def
            )
        init_op = tf.initialize_all_variables()

        sess.run(init_op)

        
