#from data.models.tutorials.image.cifar10 import cifar10,cifar10_input

import tensorflow as tf
import numpy as np
import sys
import time
import urllib
import tarfile
import os
import tensorflow_datasets as tfds

# 定义全局变量
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/nfs/syzhou/github/project/data/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# 基本模型参数
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'



# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

class DataPreprocessor(object):
  """Applies transformations to dataset record."""

  def __init__(self, distords):
    self._distords = distords

  def __call__(self, record):
    """Process img for training or eval."""
    img = record['image']
    img = tf.cast(img, tf.float32)
    if self._distords:  # training
      # Randomly crop a [height, width] section of the image.
      img = tf.random_crop(img, [IMAGE_SIZE, IMAGE_SIZE, 3])
      # Randomly flip the image horizontally.
      img = tf.image.random_flip_left_right(img)
      # Because these operations are not commutative, consider randomizing
      # the order their operation.
      # NOTE: since per_image_standardization zeros the mean and makes
      # the stddev unit, this likely has no effect see tensorflow#1458.
      img = tf.image.random_brightness(img, max_delta=63)
      img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
    else:  # Image processing for evaluation.
      # Crop the central [height, width] of the image.
      img = tf.image.resize_image_with_crop_or_pad(img, IMAGE_SIZE, IMAGE_SIZE)
    # Subtract off the mean and divide by the variance of the pixels.
    img = tf.image.per_image_standardization(img)
    return dict(input=img, target=record['label'])

def _get_images_labels(batch_size, split, distords=False):
  """Returns Dataset for given split."""
  dataset = tfds.load(name='cifar10', split=split)
  scope = 'data_augmentation' if distords else 'input'
  with tf.name_scope(scope):
    dataset = dataset.map(DataPreprocessor(distords), num_parallel_calls=10)
  # Dataset is small enough to be fully loaded on memory:
  dataset = dataset.prefetch(-1)
  dataset = dataset.repeat().batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images_labels = iterator.get_next()
  images, labels = images_labels['input'], images_labels['target']
  tf.summary.image('images', images)
  return images, labels

def distorted_inputs(batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  return _get_images_labels(batch_size, tfds.Split.TRAIN, distords=True)


def inputs(eval_data, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  split = tfds.Split.TEST if eval_data == 'test' else tfds.Split.TRAIN
  return _get_images_labels(batch_size, split)


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir # /tmp/cifar10_data
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    # 从URL中获得文件名
    filename = DATA_URL.split('/')[-1]
    # 合并文件路径
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        # 定义下载过程中打印日志的回调函数
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        # 下载数据集
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
        print()
        # 获得文件信息
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        # 解压缩
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)



# 定义卷积层
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    # input_op是输入的tensor
    # name是这一层的名字
    # kh即kernel height 卷积核的高
    # kw即kernel width 卷积核的宽
    # n_out是卷积核数量即输出通道
    # dh是步长的高
    # dw是步长的宽
    # p是参数列表
    n_in = input_op.get_shape()[-1].value  # 获取input_op的通道数

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


# 定义全连接层
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[n_in, n_out],  # [输入的通道数，输出的通道数]
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')  # 赋予0.1而不是0避免dead neuron
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation


# 定义最大池化层
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)



def print_activation(t):
    print(t.op.name,' ',t.getshape().as_list())

def inference_1(images):
    parameters=[]

    with tf.name_scope('conv1') as scope:
        kernel=tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=0.1),name='weight')
        conv=tf.nn.conv2d(images,kernel,[1,4,4,1],padding="SAME")
        biases=tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(bias,name=scope)
        print_activation(conv1)
        parameters+=[kernel,biases]
        lrn1=tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')
        pool1=tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')
        print_activation(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=0.1), name='weight')
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        print_activation(conv2)
        parameters += [kernel, biases]
        lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        print_activation(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal_initializer[3, 3, 192, 384], dtype=tf.float32, stddev=0.1,
                             name='weight')
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activation(conv3)

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal_initializer[3, 3, 384, 256], dtype=tf.float32, stddev=0.1,
                             name='weight')
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activation(conv4)

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal_initializer[3, 3, 256, 256], dtype=tf.float32, stddev=0.1,
                             name='weight')
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activation(conv5)

    pool5=tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool5')
    print_activation(pool5)
    return pool5

def inference_2(images):
    parameters=[]
    conv1=conv_op(images,name='conv1',kh=11,kw=11,n_out=64,dh=4,dw=4,p=parameters)
    lrn1=tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')
    pool1=mpool_op(lrn1, name="pool1", kh=3, kw=3, dw=2, dh=2)
    conv2=conv_op(pool1,name='conv2',kh=5,kw=5,n_out=192,dh=1,dw=1,p=parameters)
    pool2 = mpool_op(conv2, name="pool2", kh=3, kw=3, dw=2, dh=2)
    lrn2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
    conv3=conv_op(lrn2,name='conv3',kh=3,kw=3,n_out=384,dh=1,dw=1,p=parameters)
    conv4=conv_op(conv3,name='conv4',kh=3,kw=3,n_out=256,dh=1,dw=1,p=parameters)
    conv5=conv_op(conv4,name='conv5',kh=3,kw=3,n_out=10,dh=1,dw=1,p=parameters)
    pool5=mpool_op(conv5, name="pool5", kh=3, kw=3, dw=2, dh=2)
    print(pool5.shape)
    return pool5


def variable_with_weight_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var),w1,name='weightloss')#元素相乘
        tf.add_to_collection('losses',weight_loss)
    return var


def loss(logits,label):
    #print(label.shape)
    label=tf.cast(label,tf.int64)# tf.cast() 将 x 的数据格式转化成 dtype
    print(label.shape)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label,name='cross_entropy_per_example')
    print(cross_entropy.shape)
    cross_entropy_mean=tf.reduce_mean(cross_entropy,name='cross_entropy')
    print(cross_entropy_mean.shape)
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')


# 描述模型的训练
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

#train的滑动平均
def train(total_loss, global_step):
    '''
    训练 CIFAR-10模型

    创建一个optimizer并应用于所有可训练变量. 为所有可训练变量添加移动平均值.
    ARGS：
     total_loss：loss()的全部损失
     global_step：记录训练步数的整数变量
    返回：
     train_op：训练的op
    '''
    # 影响学习率的变量
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # 根据步骤数以指数方式衰减学习率
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    # Summary是对网络中Tensor取值进行监测的一种Operation.这些操作在图中是“外围”操作，不影响数据流本身.
    # 把lr添加到观测中
    tf.summary.scalar('learning_rate', lr)

    # 生成所有损失和相关和的移动平均值的summary
    loss_averages_op = _add_loss_summaries(total_loss)

    # 计算梯度
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # 应用梯度.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # 为可训练变量添加直方图summary.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # 为梯度添加直方图summary
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # 跟踪所有可训练变量的移动平均值
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    return variables_averages_op


def _add_loss_summaries(total_loss):
    '''
    往CIFAR-10模型中添加损失summary
    为所有损失和相关summary生成移动平均值，以便可视化网络的性能

    ARGS：
     total_loss：loss()的全部损失
    返回：
     loss_averages_op：用于生成移动平均的损失
    '''
    # 计算所有单个损失和总损失的移动平均
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # 把所有的单个损失和总损失添加到summary观测中，平均损失也添加观测
    for l in losses + [total_loss]:
        # 将每个损失命名为损失的原始名称+“（raw）”，并将损失的移动平均版本命名为损失的原始名称
        # 这一行代码应该已经过时了，执行时提醒：
        # INFO:tensorflow:Summary name conv1/weight_loss (raw) is illegal; using conv1/weight_loss__raw_ instead.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op




#train_op = train(loss, global_step)
#数据集的下载和解压
maybe_download_and_extract()

#处理训练数据集和测试数据集
image_train,labels_train=distorted_inputs(FLAGS.batch_size)
image_test,labels_test=inputs(True,FLAGS.batch_size)

#定义placeholder
image_holder=tf.placeholder(tf.float32,[FLAGS.batch_size,24,24,3])
label_holder=tf.placeholder(tf.int32,[FLAGS.batch_size])


#定义图
logits=inference_2(image_holder)
logits=tf.reshape(logits,(FLAGS.batch_size,10))
#定义loss
loss=loss(logits,label_holder)

#运用滑动平均模型定义trainop和testop
#train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)
global_step = tf.train.get_or_create_global_step()# tf.Variable(0, trainable=False)
train_op = train(loss, global_step)
top_k_op=tf.nn.in_top_k(logits,label_holder,1)


sess=tf.InteractiveSession()
tf.global_variables_initializer().run()

tf.train.start_queue_runners()

# 用tf.train.Saver()创建一个Saver来管理模型中的所有变量
saver = tf.train.Saver(tf.all_variables())

for step in range(FLAGS.max_steps):
    start_time=time.time()
    image_batch,label_batch=sess.run([image_train,labels_train])
    _,loss_value=sess.run([train_op,loss],feed_dict={image_holder:image_batch,label_holder:label_batch})
    duration=time.time()-start_time
    if step%10==0:
        example_per_sec=FLAGS.batch_size/duration
        sec_per_batch=float(duration)
        print('loss= ',loss_value)
    if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


num_example=10000
import math
num_iter=int(math.ceil(num_example/FLAGS.batch_size))
true_count=0
total_sample_count=num_iter*FLAGS.batch_size
step=0
while step<num_iter:
    image_batch,label_batch=sess.run([image_test,labels_test])
    prediction=sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch})
    true_count=np.num(prediction)
    step+=1
true_count=true_count/total_sample_count
print('precision',prediction)



