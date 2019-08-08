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
is_training = tf.placeholder_with_default(False, shape=(), name='training')
hidden1 = tf.layers.conv2d(X, kernel_size=5, filters=64, strides=2, padding='same', activation=tf.nn.leaky_relu)

hidden2 = tf.layers.conv2d(hidden1, kernel_size=5, filters=128, strides=2, padding='same')
hidden2 = tf.nn.leaky_relu(tf.layers.batch_normalization(hidden2, training=is_training))

hidden3 = tf.layers.conv2d(hidden2, kernel_size=5, filters=256, strides=2, padding='same')
hidden3 = tf.nn.leaky_relu(tf.layers.batch_normalization(hidden3, training=is_training))

hidden4 = tf.layers.conv2d(hidden3, kernel_size=5, filters=512, strides=2, padding='same')
hidden4 = tf.nn.leaky_relu(tf.layers.batch_normalization(hidden4, training=is_training))

hidden5 = tf.layers.flatten(hidden4)
hidden5 = tf.layers.dense(hidden5, n_hidden5_units)

hidden6_mean = tf.layers.dense(hidden5, n_hidden5_units)
hidden6_sigma = tf.layers.dense(hidden5, n_hidden5_units)
noise = tf.random_normal(tf.shape(hidden6_sigma), dtype=tf.float32)
hidden6 = hidden6_mean + hidden6_sigma * noise

hidden7 = tf.layers.dense(hidden6, n_hidden5_units)
hidden7 = tf.reshape(hidden7, [-1, 4, 4, 512])
hidden7 = tf.nn.relu(tf.layers.batch_normalization(hidden7, training=is_training))

hidden8 = tf.layers.conv2d_transpose(hidden7, kernel_size=5, filters=256, strides=2, padding='same')
hidden8 = tf.nn.relu(tf.layers.batch_normalization(hidden8, training=is_training))

hidden9 = tf.layers.conv2d_transpose(hidden8, kernel_size=5, filters=128, strides=2, padding='same')
hidden9 = tf.nn.relu(tf.layers.batch_normalization(hidden9, training=is_training))

hidden10 = tf.layers.conv2d_transpose(hidden9, kernel_size=5, filters=64, strides=2, padding='same')
hidden10 = tf.nn.relu(tf.layers.batch_normalization(hidden10, training=is_training))

logits = tf.layers.conv2d_transpose(hidden10, kernel_size=5, filters=3, strides=2, padding='same')
outputs = tf.nn.sigmoid(logits)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
reconstruction_loss = tf.reduce_sum(xentropy)
eps = 1e-10
latent_loss = 0.5 * tf.reduce_sum(
    tf.square(hidden6_sigma) + tf.square(hidden6_mean) - 1 - tf.log(eps + tf.square(hidden6_sigma))
)
sparsity_loss = tf.reduce_sum(kl_divergence(sparsity_target, tf.reduce_mean(hidden6, axis=0)))
loss = reconstruction_loss + latent_loss + sparsity_weight * sparsity_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.log_device_placement = True
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:
    def gen_dogs(bs):
        codings_rnd = np.random.normal(size=[bs, n_hidden5_units])
        outputs_val = outputs.eval(feed_dict={hidden6: codings_rnd, is_training: False})
        return outputs_val


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
            loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss],
                                                                          feed_dict={X: X_batch, is_training: True})
            print("\r{}".format(epoch), "Train total loss:", loss_val, "\tReconstruction loss:",
                  reconstruction_loss_val, "\tLatent loss:", latent_loss_val)

        # plot.
        sample_rnd = np.random.normal(size=[5, n_hidden5_units])
        sample_outputs = outputs.eval(feed_dict={hidden6: sample_rnd, is_training: False})
        plt.figure(figsize=(15, 3))
        for i, sample in enumerate(sample_outputs):
            plt.subplot(1, 5, i + 1)
            sample = np.array(sample).clip(0, 1)
            plt.axis('off')
            plt.imshow(sample)
        plt.show()

    # Generate.
    n_batches = n_generates // batch_size
    n_last_batch_size = n_generates % batch_size
    z = zipfile.PyZipFile('images.zip', mode='w')
    for i in range(n_batches):
        for j, img in enumerate(gen_dogs(batch_size)):
            f = f'sample_{i}_{j}.png'
            imsave(f, img)
            z.write(f)
            os.remove(f)

    for k, img in enumerate(gen_dogs(n_last_batch_size)):
        f = f'sample_last_{k}.png'
        imsave(f, img)
        z.write(f)
        os.remove(f)
    z.close()

print('Training end.')
