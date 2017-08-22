# ==================================
# -*- coding: utf-8
# @FileName:   crack 
# @Auther:     M.K.
# @Time:       2017/8/9-22:01
# ==================================
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from train import build_cnn_model,\
    IMAGE_WIDTH, IMAGE_HEIGHT,\
    MAX_CODE_LEN, CHAR_SETS_LEN,\
    vec2code, load_image


# load cnn model
x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
keep_prob = tf.placeholder(tf.float32)
output = build_cnn_model(x, keep_prob)
predict = tf.argmax(tf.reshape(output, [-1, MAX_CODE_LEN, CHAR_SETS_LEN]), 2)
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint("./models/model2"))


def crack_security_code(image_loc):
    """
    crack security code with cnn
    :param image_loc:
    :return:
    """
    # load image
    _img = Image.open(image_loc)
    img = load_image(image_loc)

    # predict security code
    vec = sess.run(predict, feed_dict={
        x: [img], keep_prob: 1.0
    })
    vec = vec[0].tolist()

    return vec2code(vec)


def show_imgage(image_loc):
    image = Image.open(image_loc)
    code = crack_security_code(image_loc)

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.5, 0.9, code, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    from os import listdir, path
    import random

    download = './download'
    tagged = './tagged'

    # count = 0
    all_images = listdir(download)
    random.shuffle(all_images)
    for img in all_images:
        show_imgage(path.join(download, img))
        # idx, _ = img.split('.')
        #
        # src = path.join(download, img)
        # try:
        #     r_img = Image.open(src)
        # except OSError:
        #     os.remove(src)
        #     continue
        #
        # predict_code = crack_security_code(path.join(download, img))
        # dst = path.join(tagged, idx + "." + predict_code + ".png")
        # r_img.save(dst)









