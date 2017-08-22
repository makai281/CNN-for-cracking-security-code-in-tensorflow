# ==================================
# -*- coding: utf-8
# @FileName:   train 
# @Auther:     M.K.
# @Time:       2017/8/9-22:00
# ==================================
import os
import string

from PIL import Image
import numpy as np
import tensorflow as tf


CHAR_SETS = list(string.ascii_lowercase) + list(string.digits)
IMAGE_WIDTH = 120
IMAGE_HEIGHT = 50
CHAR_SETS_LEN = len(CHAR_SETS)
MAX_CODE_LEN = 5

idx2char = {}
char2idx = {}
for idx, char in enumerate(CHAR_SETS):
    idx2char[idx] = char
    char2idx[char] = idx


def code2vec(code):
    """
    convert security code into vector
    :param code:
    :return:
    """
    code_len = len(code)
    # print(code)
    if code_len > MAX_CODE_LEN:
        raise ValueError('验证码最长5个字符')

    mat = np.zeros((MAX_CODE_LEN, CHAR_SETS_LEN))

    for i, c in enumerate(code):
        idx = (i, char2idx[c])
        mat[idx] = 1

    return mat.flatten()


def vec2code(vec):
    """
    convert vector into security code
    :param vec:
    :return:
    """
    text = []
    for c in vec:
        _idx = c % CHAR_SETS_LEN
        _char = idx2char[_idx]
        text.append(_char)
    return "".join(text)


def convert2gray(img):
    """
    convert colorful image into gray image
    :param img:
    :return:
    """
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def load_image(img_loc, box=[40, 0, 160, 50]):
    """
    read and trim image
    :param img_loc:
    :param box:
    :return: flattened image vector
    """
    # read image
    img = Image.open(img_loc)

    # trim image
    img = img.crop(box)

    # conver image into numpy array
    img = np.asarray(img)

    # convert color image to gray image
    image = convert2gray(img)

    return image.flatten() / 255


def generate_batch_images(path, batch_size=128):
    """
    generate batch samples
    :param path:
    :param batch_size:
    :return:
    """
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CODE_LEN * CHAR_SETS_LEN])

    count = 0
    for i, img in enumerate(os.listdir(path)):
        abs_img = os.path.join(path, img)
        # print(img)
        _code = img.split('.')[1]
        image = load_image(abs_img)

        try:
            batch_y[i % batch_size, :] = code2vec(_code.lower())
        except:
            # print(img)
            continue

        batch_x[i % batch_size, :] = image
        count += i
        if count % batch_size == 0:
            yield batch_x, batch_y
            batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
            batch_y = np.zeros([batch_size, MAX_CODE_LEN * CHAR_SETS_LEN])

    yield batch_x, batch_y


def weight_variable(shape):
    """
    weight_variable generates a weight variable of a given shape.
    :param shape:
    :return:
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    bias_variable generates a bias variable of a given shape.
    :param shape:
    :return:
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    conv2d returns a 2d convolution layer with full stride.
    :param x:
    :param W:
    :return:
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    """
    max_pool_2x2 downsamples a feature map by 2X.
    :param x:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def build_cnn_model(x, keep_prob):
    """
    build cnn model for classifying security code
    :param x: an input tensor with the dimensions (batch_size, IMAGE_HEIGHT * IMAGE_WIDTH)
    :param keep_prob: dropout rate
    :return:
    """
    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # first convolutional layer
    W_conv1 = weight_variable([3, 5, 1, 8])
    b_conv1 = bias_variable([8])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # pooling layer - downsamples by 2x
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = weight_variable([5, 3, 8, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # second pooling layer
    h_pool2 = max_pool_2x2(h_conv2)

    # third convolutional layer
    W_conv3 = weight_variable([3, 5, 16, 32])
    b_conv3 = bias_variable([32])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    # third pooling layer
    h_pool3 = max_pool_2x2(h_conv3)

    # fully connected layer 1
    W_fc1 = weight_variable([4 * 12 * 32, 256])
    b_fc1 = bias_variable([256])

    h_pool2_flat = tf.reshape(h_pool3, [-1, 4 * 12 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout layer
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # fully connected layer 2
    W_fc2 = weight_variable([256, MAX_CODE_LEN * CHAR_SETS_LEN])
    b_fc2 = bias_variable([MAX_CODE_LEN * CHAR_SETS_LEN])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv


def run(epochs=100):
    # create the model
    x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    keep_prob = tf.placeholder(tf.float32)


    # define the loss and optimizer
    y = tf.placeholder(tf.float32, [None, MAX_CODE_LEN * CHAR_SETS_LEN])

    # build the graph for the cnn
    y_conv = build_cnn_model(x, keep_prob)

    # define the loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y))
    optimizer_adam = tf.train.AdamOptimizer().minimize(loss)

    lr = tf.placeholder(dtype=tf.float32, name='lr')
    optimizer_sgd = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    predict = tf.reshape(y_conv, [-1, MAX_CODE_LEN, CHAR_SETS_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(y, [-1, MAX_CODE_LEN, CHAR_SETS_LEN]), 2)

    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        best_dev = -np.inf
        l_r = 0.01
        t = 0
        for e in range(epochs):
            print('starting epoch %d...' % e)
            count = 0
            for i, (batch_x_train, batch_y_train) in enumerate(generate_batch_images('./images/train')):
                count += 1
                if e < 10:
                    optimizer_adam.run(feed_dict={
                        x: batch_x_train, y: batch_y_train, keep_prob: 0.5
                    })
                else:
                    optimizer_sgd.run(feed_dict={
                        x: batch_x_train, y: batch_y_train, lr: l_r, keep_prob: 0.5
                    })
                if count % 10 == 0:
                    # compute accuracy on dev
                    total_acc = []
                    for batch_x_dev, batch_y_dev in generate_batch_images('./images/dev'):
                        dev_accuracy = accuracy.eval(feed_dict={
                            x: batch_x_dev, y: batch_y_dev, keep_prob: 1.0
                        })
                        total_acc.append(dev_accuracy)
                    if best_dev < np.mean(total_acc):
                        best_dev = np.mean(total_acc)
                        print('New best dev accuracy %f...' % best_dev)
                        saver.save(sess, './models/model3/crack_security_code.model')
                    elif e >= 10:
                        t += 1
                        l_r = l_r / np.sqrt(t)
                    print('step %d, dev accuracy %f' % (count, np.mean(total_acc)))
            # compute accuracy on test
            acc = []
            for batch_x_test, batch_y_test in generate_batch_images('./images/test'):
                test_accuracy = accuracy.eval(feed_dict={
                    x: batch_x_test, y: batch_y_test, keep_prob: 1.0
                })
                acc.append(test_accuracy)

            print('epoch %d, test accuracy %f' % (e, np.mean(acc)))


if __name__ == "__main__":
    run(200)





