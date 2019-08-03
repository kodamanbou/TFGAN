import glob
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, concatenate, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, Adam
import numpy as np
import os
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt

root_images = "../input/all-dogs/all-dogs/"
root_annots = "../input/annotation/Annotation/"
OUTPUT_DIR = 'samples_dogs'
GEN_DIR = 'generated_dogs'

all_images = os.listdir(root_images)


def prepro():
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if not os.path.exists(GEN_DIR):
        os.mkdir(GEN_DIR)

    breeds = glob.glob(root_annots + '*')
    annotation = []

    for b in breeds:
        annotation += glob.glob(b + '/*')

    map = {}
    for annot in annotation:
        breed = annot.split('/')[-2]
        index = breed.split('-')[0]
        map.setdefault(index, breed)

    print(f'Total breeds：{len(map)}')
    return map


breed_map = prepro()


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


# preprocess data.
imagesIn = []
for img in all_images:
    imagesIn.append(read_image(img, 64, 64))
imagesIn = np.array(imagesIn)
print('imagesIn_shape: ', imagesIn.shape)

dog = Input((12288,))
dog_name = Input((10000,))
x = Dense(12288, activation='sigmoid')(dog_name)
x = Reshape((2, 12288, 1))(concatenate([dog, x]))
x = Conv2D(1, (2, 1), use_bias=False, name='conv')(x)
discriminated = Flatten()(x)

discriminator = Model([dog, dog_name], discriminated)
discriminator.get_layer('conv').trainable = False
discriminator.get_layer('conv').set_weights([np.array([[[[-1.0]]], [[[1.0]]]])])
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

discriminator.summary()

# Train Discriminator.
train_y = imagesIn[:10000, :, :, :].reshape((-1, 12288))
train_x = np.zeros((10000, 10000))
for i in range(10000):
    train_x[i, i] = 1
zeros = np.zeros((10000, 12288))

lr = 0.5
for k in range(5):
    annealer = LearningRateScheduler(lambda x: lr)
    h = discriminator.fit([zeros, train_x], train_y, epochs=10, batch_size=256,
                          callbacks=[annealer], verbose=0)
    print('Epoch', (k + 1) * 10, '/30- loss=', h.history['loss'][-1])
    if h.history['loss'][-1] < 0.533:
        lr = 0.1

# Delete input data.
del train_x, train_y, imagesIn

# Bad Memory.
seed = Input((10000,))
x = Dense(2048, activation='elu')(seed)
x = Reshape((8, 8, 32))(x)
x = Conv2D(128, (3, 3), activation='elu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='elu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='elu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), activation='linear', padding='same')(x)
generated = Flatten()(x)

generator = Model(seed, [generated, Reshape((10000,))(seed)])
generator.summary()

# Build GAN.
discriminator.trainable = False
gan_input = Input(shape=(10000,))
x = generator(gan_input)
gan_output = discriminator(x)

gan = Model(gan_input, gan_output)
gan.get_layer('model_1').get_layer('conv').set_weights([np.array([[[[-1]]], [[[255.]]]])])
gan.compile(optimizer=Adam(5), loss='mean_squared_error')
gan.summary()

# Training data.
train = np.zeros((10000, 10000))
for i in range(10000):
    train[i, i] = 1
zeros = np.zeros((10000, 12288))

ep = 1.
it = 9
lr = 5e-3

for k in range(it):
    annealer = LearningRateScheduler(lambda x: lr)
    h = gan.fit(train, zeros, batch_size=256, epochs=ep,
                callbacks=[annealer], verbose=0)
    print('loss: ', h.history['loss'][-1])
    plt.figure(figsize=(15, 3))
    for j in range(5):
        xx = np.zeros(10000)
        xx[np.random.randint(10000)] = 1
        plt.subplot(1, 5, j + 1)
        img = generator.predict(xx.reshape((-1, 10000)))[0].reshape((-1, 64, 64, 3))
        img = Image.fromarray(img.astype('uint8').reshape((64, 64, 3)))
        plt.axis('off')
        plt.imshow(img)
    plt.show()

    ep *= 2
    if ep >= 32: lr = 0.001
    if ep > 256: ep = 256
    ep *= 2
    if ep >= 32: lr = 0.001
    if ep > 256: ep = 256
