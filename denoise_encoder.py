import tensorflow as tf
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import zipfile
from imageio import imsave
import matplotlib.pyplot as plt

image_size = 64
n_epochs = 10
batch_size = 128
learning_rate = 0.01
dropout_rate = 0.3
n_generates = 10000
n_hidden_units = 4 * 4 * 512

root_images = "../input/all-dogs/all-dogs/"
root_annots = "../input/annotation/Annotation/"
all_images = os.listdir(root_images)


def prepro():
    breed_map = {}
    breeds = glob.glob(root_annots + '*')
    annotation = []

    for b in breeds:
        annotation += glob.glob(b + '/*')

    for annot in annotation:
        breed = annot.split('/')[-2]
        index = breed.split('-')[0]
        breed_map.setdefault(index, breed)

    print(f'Total breeds：{len(breed_map)}')
    return breed_map


def read_image(image_name, height, width):
    image = Image.open(os.path.join(root_images, image_name))
    image = np.array(image.resize((height, width)))
    return image / 255.


def _parse_fn(filename):
    breed_map = prepro()
    bpath = root_annots + str(breed_map[filename.split("_")[0]]) + "/" + str(filename.split(".")[0])
    tree = ET.parse(bpath)
    root = tree.getroot()
    objects = root.findall('object')
    for o in objects:
        bndbox = o.find('bndbox')  # reading bound box
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

    bbox = (xmin, ymin, xmax, ymax)

    image = tf.image.decode_jpeg(tf.read_file(root_images + filename))
    image = tf.image.crop_and_resize(image, bbox, crop_size=(64, 64))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    image = tf.image.rot90(image, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    return tf.cast(image, tf.float32)


dataset = tf.data.Dataset.from_tensor_slices(all_images)
dataset = dataset.map(_parse_fn)
dataset = dataset.batch(batch_size)
dataset = dataset.shuffle(1000)
iterator = dataset.make_initializable_iterator()
iter_init_op = iterator.initializer

X = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
is_training = tf.placeholder_with_default(False, shape=())
X_drop = tf.layers.dropout(X, dropout_rate, training=is_training)
conv1 = tf.layers.conv2d(X_drop, kernel_size=5, filters=64, strides=1, padding='same', activation=tf.nn.leaky_relu)

pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=2, padding='same')

conv2 = tf.layers.conv2d(pool1, kernel_size=5, filters=128, strides=1, padding='same')
conv2 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv2, training=is_training))

pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=2, padding='same')

conv3 = tf.layers.conv2d(pool2, kernel_size=5, filters=256, strides=1, padding='same')
conv3 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv3, training=is_training))

pool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=2, padding='same')

conv4 = tf.layers.conv2d(pool3, kernel_size=5, filters=512, strides=1, padding='same')
conv4 = tf.nn.leaky_relu(tf.layers.batch_normalization(conv4, training=is_training))

pool4 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=2, padding='same')

hidden = tf.layers.flatten(pool4)
hidden = tf.layers.dense(hidden, n_hidden_units)

reshape = tf.reshape(hidden, [-1, 4, 4, 512])
reshape = tf.nn.relu(tf.layers.batch_normalization(reshape, training=is_training))

deconv1 = tf.layers.conv2d_transpose(reshape, kernel_size=5, filters=256, strides=2, padding='same')
deconv1 = tf.nn.relu(tf.layers.batch_normalization(deconv1, training=is_training))

deconv2 = tf.layers.conv2d_transpose(deconv1, kernel_size=5, filters=128, strides=2, padding='same')
deconv2 = tf.nn.relu(tf.layers.batch_normalization(deconv2, training=is_training))

deconv3 = tf.layers.conv2d_transpose(deconv2, kernel_size=5, filters=64, strides=2, padding='same')
deconv3 = tf.nn.relu(tf.layers.batch_normalization(deconv3, training=is_training))

outputs = tf.layers.conv2d_transpose(deconv3, kernel_size=5, filters=3, strides=2, padding='same',
                                     activation=tf.nn.relu)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.log_device_placement = True
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:
    init.run()
    for epoch in range(n_epochs):
        sess.run(iter_init_op)
        loss_val = 0
        X_batch = None
        while True:
            try:
                X_batch = iterator.get_next()
                sess.run(training_op, feed_dict={X: X_batch, is_training: True})
                loss_val = reconstruction_loss.eval(feed_dict={X: X_batch})
            except tf.errors.OutOfRangeError:
                break

        print('Epoch: ', epoch, '\tLoss: ', loss_val)

        X_test = X_batch[:5]
        samples = outputs.eval(feed_dict={X: X_test})
        plt.figure(figsize=(15, 3))
        for i, img in enumerate(samples):
            plt.subplot(1, 5, i + 1)
            img = np.array(img).clip(0, 1)
            plt.axis('off')
            plt.imshow(img)
        plt.show()
