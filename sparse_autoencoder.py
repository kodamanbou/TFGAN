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
n_epochs = 100
batch_size = 256
learning_rate = 0.001
sparsity_target = 0.1
sparsity_weight = 0.2
n_generates = 10000
n_hidden5_units = 4 * 4 * 512

root_images = "../input/all-dogs/all-dogs/"
root_annots = "../input/annotation/Annotation/"
all_images = os.listdir(root_images)
breed_map = {}


def prepro():
    breeds = glob.glob(root_annots + '*')
    annotation = []

    for b in breeds:
        annotation += glob.glob(b + '/*')

    for annot in annotation:
        breed = annot.split('/')[-2]
        index = breed.split('-')[0]
        breed_map.setdefault(index, breed)

    print(f'Total breeds：{len(breed_map)}')


def read_image(image_name, height, width):
    bpath = root_annots + str(breed_map[image_name.split("_")[0]]) + "/" + str(image_name.split(".")[0])
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

    image = Image.open(os.path.join(root_images, image_name))
    image = image.crop(bbox)
    image = np.array(image.resize((height, width)))
    return image / 255.


def kl_divergence(p, q):
    return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))


X = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
is_training = tf.placeholder(tf.bool, name='is_training')
hidden1 = tf.layers.conv2d(X, kernel_size=5, filters=64, strides=2, padding='same', activation=tf.nn.leaky_relu)

hidden2 = tf.layers.conv2d(hidden1, kernel_size=5, filters=128, strides=2, padding='same')
hidden2 = tf.nn.leaky_relu(tf.layers.batch_normalization(hidden2, training=is_training))

hidden3 = tf.layers.conv2d(hidden2, kernel_size=5, filters=256, strides=2, padding='same')
hidden3 = tf.nn.leaky_relu(tf.layers.batch_normalization(hidden3, training=is_training))

hidden4 = tf.layers.conv2d(hidden3, kernel_size=5, filters=512, strides=2, padding='same')
hidden4 = tf.nn.leaky_relu(tf.layers.batch_normalization(hidden4, training=is_training))

hidden5 = tf.layers.flatten(hidden4)
hidden5 = tf.layers.dense(hidden5, n_hidden5_units, activation=tf.nn.sigmoid)

hidden6 = tf.reshape(hidden5, [-1, 4, 4, 512])
hidden6 = tf.nn.relu(tf.layers.batch_normalization(hidden6, training=is_training))

hidden7 = tf.layers.conv2d_transpose(hidden6, kernel_size=5, filters=256, strides=2, padding='same')
hidden7 = tf.nn.relu(tf.layers.batch_normalization(hidden7, training=is_training))

hidden8 = tf.layers.conv2d_transpose(hidden7, kernel_size=5, filters=128, strides=2, padding='same')
hidden8 = tf.nn.relu(tf.layers.batch_normalization(hidden8, training=is_training))

hidden9 = tf.layers.conv2d_transpose(hidden8, kernel_size=5, filters=64, strides=2, padding='same')
hidden9 = tf.nn.relu(tf.layers.batch_normalization(hidden9, training=is_training))

logits = tf.layers.conv2d_transpose(hidden9, kernel_size=5, filters=3, strides=2, padding='same')
outputs = tf.nn.sigmoid(logits)

hidden5_mean = tf.reduce_mean(hidden5, axis=0)
sparsity_loss = tf.reduce_sum(kl_divergence(sparsity_target, hidden5_mean))
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
loss = reconstruction_loss + sparsity_weight * sparsity_loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.log_device_placement = True
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:
    print('Training start.')
    prepro()
    init.run()
    for epoch in range(n_epochs):
        n_batches = len(all_images) // batch_size
        offset = 0
        for iteration in range(n_batches):
            X_batch = np.array([
                read_image(img, image_size, image_size)
                for img in all_images[offset:offset + batch_size]
            ])
            sess.run(training_op, feed_dict={X: X_batch, is_training: True})
            loss_val, reconstruction_loss_val, sparsity_loss_val = sess.run([loss, reconstruction_loss, sparsity_loss],
                                                                            feed_dict={X: X_batch, is_training: True})
            print("\r{}".format(epoch), "Train total loss:", loss_val, "\tReconstruction loss:",
                  reconstruction_loss_val, "\tSparsity loss:", sparsity_loss_val)

        # plot.
        X_test = np.array([
            read_image(img, image_size, image_size)
            for img in all_images[:5]
        ])
        sample_outputs = outputs.eval(feed_dict={X: X_test, is_training: False})
        plt.figure(figsize=(15, 3))
        for i, sample in enumerate(sample_outputs):
            plt.subplot(1, 5, i + 1)
            sample = np.array(sample).clip(0, 1)
            plt.axis('off')
            plt.imshow(sample)
        plt.show()

print('Training end.')
