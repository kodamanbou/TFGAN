import tensorflow as tf
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import random

image_size = 64
n_epochs = 300
batch_size = 256
learning_rate = 0.001
dropout_rate = 0.3
n_generates = 10000
n_hidden_units = 4 * 4 * 512

root_images = "../input/all-dogs/all-dogs/"
root_annots = "../input/annotation/Annotation/"
all_image_paths = os.listdir(root_images)


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


def get_bounding_cropped():
    images = []
    breed_map = prepro()
    for filename in all_image_paths:
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

        bbox = [ymin, xmin, ymax, xmax]
        image = Image.open(root_images + filename)
        images.append(np.asarray(image))

    return np.asarray(images)


def flip(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x


def color(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


def rotate(x: tf.Tensor) -> tf.Tensor:
    return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def zoom(x: tf.Tensor) -> tf.Tensor:
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))


data = (get_bounding_cropped() / 255.).astype(np.float32)
print('data_shape: ', data.shape)
dataset = tf.data.Dataset.from_tensor_slices(data)
augmentations = [flip, color, zoom, rotate]

for f in augmentations:
    dataset = dataset.map(lambda x: tf.cond(tf.random_uniform([], 0, 1) > 0.75, lambda: f(x), lambda: x),
                          num_parallel_calls=4)
dataset = dataset.map(lambda x: tf.clip_by_value(x, 0, 1))
dataset = dataset.batch(batch_size)
dataset = dataset.shuffle(1000)

iterator = dataset.make_initializable_iterator()
data_init_op = iterator.initializer

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
        sess.run(data_init_op)
        while True:
            try:
                X_batch = iterator.get_next()
                sess.run(training_op, feed_dict={X: X_batch, is_training: True})
            except tf.errors.OutOfRangeError:
                break
