import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from imageio import imsave
import glob
from PIL import Image
import xml.etree.ElementTree as ET
import zipfile

IMAGE_SIZE = 64
LOGDIR = 'logdir'
root_images = "../input/all-dogs/all-dogs/"
root_annots = "../input/annotation/Annotation/"

all_images = os.listdir(root_images)

num_iteration = 1000000
batch_size = 32
sample_size = 5
z_dim = 100
num_generate = 10000

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


def bounding_box(image):
    bpath = root_annots + str(breed_map[image.split("_")[0]]) + "/" + str(image.split(".")[0])
    tree = ET.parse(bpath)
    root = tree.getroot()
    objects = root.findall('object')
    for o in objects:
        bndbox = o.find('bndbox')  # reading bound box
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

    return (xmin, ymin, xmax, ymax)


def get_crop_image(image):
    bbox = bounding_box(image)
    im = Image.open(os.path.join(root_images, image))
    im = im.crop(bbox)
    return im


def read_image(image_name, height, width):
    image = get_crop_image(image_name)

    h = image.size[0]
    w = image.size[1]

    image = np.array(image.resize((height, width)))
    return image / 255.


def generator(z, is_training=False):
    momentum = 0.9
    with tf.variable_scope('generator'):
        h0 = tf.layers.dense(z, units=4 * 4 * 512)
        h0 = tf.reshape(h0, shape=[-1, 4, 4, 512])
        h0 = tf.nn.relu(tf.layers.batch_normalization(h0, training=is_training, momentum=momentum))

        h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=256, strides=2, padding='same')
        h1 = tf.nn.relu(tf.layers.batch_normalization(h1, training=is_training, momentum=momentum))

        h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=128, strides=2, padding='same')
        h2 = tf.nn.relu(tf.layers.batch_normalization(h2, training=is_training, momentum=momentum))

        h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=64, strides=2, padding='same')
        h3 = tf.nn.relu(tf.layers.batch_normalization(h3, training=is_training, momentum=momentum))

        h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=3, strides=2, padding='same',
                                        activation=tf.nn.tanh, name='g')
        return h4


def discriminaor(image, reuse=None, is_training=False):
    momentum = 0.9
    with tf.variable_scope('discriminator', reuse=reuse):
        h0 = tf.nn.leaky_relu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2,
                                               padding='same'))

        h1 = tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same')
        h1 = tf.nn.leaky_relu(tf.layers.batch_normalization(h1, training=is_training, momentum=momentum))

        h2 = tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same')
        h2 = tf.nn.leaky_relu(tf.layers.batch_normalization(h2, training=is_training, momentum=momentum))

        h3 = tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same')
        h3 = tf.nn.leaky_relu(tf.layers.batch_normalization(h3, training=is_training, momentum=momentum))

        h4 = tf.layers.flatten(h3)
        h4 = tf.layers.dense(h4, units=1)
        return tf.nn.sigmoid(h4), h4


class Graph:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='X')
        self.z = tf.placeholder(tf.float32, shape=[None, z_dim], name='noise')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.g_outputs = generator(self.z, self.is_training)
        self.d_real, self.d_real_logits = discriminaor(self.x, is_training=self.is_training)
        self.d_fake, self.d_fake_logits = discriminaor(self.g_outputs, reuse=True)

        self.vars_g = [var for var in tf.trainable_variables()
                       if var.name.startswith('generator')]
        self.vars_d = [var for var in tf.trainable_variables()
                       if var.name.startswith('discriminator')]

        self.d_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_real), logits=self.d_real_logits)
        )
        self.d_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.d_fake), logits=self.d_fake_logits)
        )
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_fake), logits=self.d_fake_logits)
        )

        self.d_loss = self.d_real_loss + self.d_fake_loss

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.d_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5) \
                .minimize(self.d_loss, var_list=self.vars_d, global_step=self.global_step)
            self.g_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5) \
                .minimize(self.g_loss, var_list=self.vars_g)


if __name__ == '__main__':
    print('Training start.')
    g = Graph()
    prepro()

    config = tf.ConfigProto()
    config.log_device_placement = True
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    sv = tf.train.Supervisor()
    saver = sv.saver

    with sv.managed_session(config=config) as sess:
        def generate_dogs(bs):
            n = np.random.uniform(-1.0, 1.0, [bs, z_dim]).astype(np.float32)
            gen_imgs = sess.run(g.g_outputs, feed_dict={g.z: n, g.is_training: False})
            gen_imgs = (gen_imgs + 1) / 2
            return gen_imgs


        loss = {'d': [], 'g': []}
        z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)

        offset = -batch_size

        for i in tqdm(range(num_iteration)):
            n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)

            offset = (offset + batch_size) % len(all_images)
            batch = np.array(
                [read_image(img, IMAGE_SIZE, IMAGE_SIZE)
                 for img in all_images[offset:offset + batch_size]])
            batch = (batch - 0.5) * 2  # batch regularize.

            d_ls, g_ls = sess.run([g.d_loss, g.g_loss], feed_dict={g.x: batch, g.z: n, g.is_training: True})
            loss['d'].append(d_ls)
            loss['g'].append(g_ls)

            gs = sess.run(g.global_step)
            sess.run(g.d_train_op, feed_dict={g.x: batch, g.z: n, g.is_training: True})
            sess.run(g.g_train_op, feed_dict={g.x: batch, g.z: n, g.is_training: True})

            if gs % 10000 == 0:
                print(gs, d_ls, g_ls)
                plt.figure(figsize=(15, 3))
                gen_img = sess.run(g.g_outputs, feed_dict={g.z: z_samples, g.is_training: False})
                gen_img = (gen_img + 1) / 2
                for j, img in enumerate(gen_img[:sample_size]):
                    plt.subplot(1, sample_size, j + 1)
                    img = np.array(img).clip(0, 1)
                    plt.axis('off')
                    plt.imshow(img)
                plt.show()

        plt.plot(loss['d'], label='Discriminator')
        plt.plot(loss['g'], label='Generator')
        plt.legend(loc='upper right')
        plt.show()

        num_batch = num_generate // batch_size
        last_batch_size = num_generate % batch_size
        z = zipfile.PyZipFile('images.zip', mode='w')

        for i, img in tqdm(enumerate(generate_dogs(batch_size))):
            f = f'sample_{i}.png'
            imsave(f, img)
            z.write(f)
            os.remove(f)

        for j, img in tqdm(enumerate(generate_dogs(last_batch_size))):
            f = f'sample_last_{j}.png'
            imsave(f, img)
            z.write(f)
            os.remove(f)

        z.close()

    print('Training end.')
