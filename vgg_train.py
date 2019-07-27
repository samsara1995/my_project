import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tf_vgg
import numpy as np

INPUT_NODE=784
OUTPUT_NODE=10

IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10

CONV1_DEEP=32
CONV1_SIZE=5

CONV2_DEEP=64
CONV2_SIZE=5

FC_SIZE=512

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "/nfs/syzhou/github/project/model/"
MODEL_NAME = "model.ckpt"


def train(mnist):
    with tf.name_scope('input'):#处理输入的都放在input下面
        x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS], name="X-input")
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    '''
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size, 3],
                                              dtype=tf.float32,
                                              stddev=1e-1))
    '''
    #keep_prob = tf.placeholder(tf.float32)

    predictions, softmax, fc8, p = tf_vgg.inference_op(x, keep_prob=1.0)

    #print('y shape',y.shape)
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope("loss_funtion"):
        print('sotf    ',softmax.shape)
        print('label   ',tf.argmax(y_, 1).shape)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=softmax, labels=tf.argmax(y_, 1))
        #print(cross_entropy)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        #loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
        loss=cross_entropy_mean

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
            reshaped_xs=np.reshape(xs,(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
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
