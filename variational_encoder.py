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
batch_size = 8
learning_rate = 0.001
n_generates = 10000
n_hidden_units = 4096

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


X = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
is_training = tf.placeholder(tf.bool, name='is_training')
conv1 = tf.layers.conv2d(X, kernel_size=5, filters=64, strides=1, padding='same', activation=tf.nn.leaky_relu)

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
hidden_mean = tf.layers.dense(hidden, n_hidden_units)
hidden_sigma = tf.layers.dense(hidden, n_hidden_units)
noise = tf.random_normal(tf.shape(hidden_sigma), dtype=tf.float32)
hidden = hidden_mean + hidden_sigma * noise

hidden2 = tf.layers.dense(hidden, 4 * 4 * 512, activation=tf.nn.elu)

reshape = tf.reshape(hidden2, [-1, 4, 4, 512])
reshape = tf.nn.relu(tf.layers.batch_normalization(reshape, training=is_training))

deconv1 = tf.layers.conv2d_transpose(reshape, kernel_size=5, filters=256, strides=2, padding='same')
deconv1 = tf.nn.relu(tf.layers.batch_normalization(deconv1, training=is_training))

deconv2 = tf.layers.conv2d_transpose(deconv1, kernel_size=5, filters=128, strides=2, padding='same')
deconv2 = tf.nn.relu(tf.layers.batch_normalization(deconv2, training=is_training))

deconv3 = tf.layers.conv2d_transpose(deconv2, kernel_size=5, filters=64, strides=2, padding='same')
deconv3 = tf.nn.relu(tf.layers.batch_normalization(deconv3, training=is_training))

logits = tf.layers.conv2d_transpose(deconv3, kernel_size=5, filters=3, strides=2, padding='same')
outputs = tf.nn.sigmoid(logits)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
reconstruction_loss = tf.reduce_sum(xentropy)
eps = 1e-10
latent_loss = 0.5 * tf.reduce_sum(
    tf.square(hidden_sigma) + tf.square(hidden_mean) - 1 - tf.log(eps + tf.square(hidden_sigma))
)
loss = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.log_device_placement = True
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:
    def gen_dogs(bs):
        codings_rnd = np.random.normal(size=[bs, n_hidden_units])
        outputs_val = outputs.eval(feed_dict={hidden: codings_rnd, is_training: False})
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
        sample_rnd = np.random.normal(size=[5, n_hidden_units])
        sample_outputs = outputs.eval(feed_dict={hidden: sample_rnd, is_training: False})
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
