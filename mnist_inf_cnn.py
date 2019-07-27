import tensorflow as tf

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


def inference(input_tensor,train,regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights=tf.get_variable(
            "weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_biases=tf.get_variable(
            "bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0)
        )
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
        shape = relu1.get_shape().as_list()
        #print('conv1 shape',shape)
    #变量名就是指针，变量的name属性是地址

    with tf.name_scope('layer2-pool1'):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #因为这里没有定义weight,bias等内容，所以不用产生新的variable范围，故这里用name_scope
        #print('pool1 shape',pool1.get_shape().as_list())

    with tf.variable_scope('layer3-conv2'):
        conv2_weights=tf.get_variable(
            "weight",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases=tf.get_variable(
            "bias",[CONV2_DEEP],initializer=tf.constant_initializer(0.0)
        )
        conv2=tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #7*7*64
    pool_shape=pool2.get_shape().as_list()#可见只能用于tensor来返回shape，但是是一个元组，需要通过as_list()的操作转换成list
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshapeed=tf.reshape(pool2,[pool_shape[0],nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights=tf.get_variable(
            "weight",[nodes,FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases=tf.get_variable(
            "bias",[FC_SIZE],initializer=tf.constant_initializer(0.1)
        )
        fc1=tf.nn.relu(tf.matmul(reshapeed,fc1_weights)+fc1_biases)
        if train:fc1=tf.nn.dropout(fc1,0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights=tf.get_variable(
            "weight",[FC_SIZE,NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases=tf.get_variable(
            "bias",[NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        logit=tf.matmul(fc1,fc2_weights)+fc2_biases

    return logit

