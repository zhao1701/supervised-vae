import os
import sys
import pytest
import shutil

import numpy as np

sys.path.append('..')

from svae_mnist import SVAE
from keras.utils import to_categorical
from skimage.transform import resize
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NUM_LATENTS = 32
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 1
CHECKPOINT_DIR = '../experiments/experiment-mnist-02/checkpoints/'
LOG_DIR = '../experiments/experiment-mnist-02/logs/'

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
BETA = 1

directories = [CHECKPOINT_DIR, LOG_DIR]
for directory in directories:
	if not os.path.isdir(directory):
		os.makedirs(directory)

def make_svae():
	svae = SVAE(
		CHECKPOINT_DIR, LOG_DIR,
		img_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
		num_latents=32, num_classes=10)
	return svae

svae = make_svae()

def resize_data(X):
  image_list = list()
  for image in X:
    image_list.append(resize(image, output_shape=(32, 32)))
  X = np.array(image_list)
  # print(X.shape)
  # print(len(image_list))
  assert(X.shape == (len(image_list), 32, 32, 1))
  return X

datagen = ImageDataGenerator()

mnist = input_data.read_data_sets('../data/MNIST/', reshape=False)
X_train = mnist.train.images
X_train = resize_data(X_train)
y_train = mnist.train.labels

X_val = mnist.validation.images
X_val = resize_data(X_val)
y_val = mnist.validation.labels

train_datagen = datagen.flow(X_train, y_train)
validation_datagen = datagen.flow(X_val, y_val)

svae.fit_classifier_generator(
	train_datagen, validation_datagen,
	learning_rate=LEARNING_RATE,
	beta=BETA,
	num_epochs=20)

svae.fit_decoder_generator(
	train_datagen, validation_datagen,
	learning_rate=LEARNING_RATE,
	num_epochs=20)