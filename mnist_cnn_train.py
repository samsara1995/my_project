import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inf_cnn
import numpy as np
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "/nfs/syzhou/github/project/model/"
MODEL_NAME = "model.ckpt"


def train(mnist):
    with tf.name_scope('input'):#处理输入的都放在input下面
        x = tf.placeholder(tf.float32, [BATCH_SIZE, mnist_inf_cnn.IMAGE_SIZE,mnist_inf_cnn.IMAGE_SIZE,mnist_inf_cnn.NUM_CHANNELS], name="X-input")
        y_ = tf.placeholder(tf.float32, [None, mnist_inf_cnn.OUTPUT_NODE], name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inf_cnn.inference(x,1 ,regularizer)
    print('y shape',y.shape)#100 10
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope("loss_funtion"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        print(cross_entropy)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                                   LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)


    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name="train")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs=np.reshape(xs,(BATCH_SIZE,mnist_inf_cnn.IMAGE_SIZE,mnist_inf_cnn.IMAGE_SIZE,mnist_inf_cnn.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 1000 == 0:
                print("after %d training steps,batch is %g" % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

    writer=tf.summary.FileWriter('/nfs/syzhou/github/project/path/to/log',tf.get_default_graph())
    writer.close()


def main(argv=None):
    mnist = input_data.read_data_sets("/nfs/syzhou/github/project/tmp/data/", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
