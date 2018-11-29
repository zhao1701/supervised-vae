import os
import sys
import csv
import shutil
import glob
import imageio
import warnings

import numpy as np
import argparse as ap

sys.path.append('..')
warnings.filterwarnings('ignore')

from svae_mnist import SVAE
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.examples.tutorials.mnist import input_data

EXPERIMENT_DIR = '../experiments/experiment-mnist-02/'
NUM_LATENTS = 32
NUM_ROWS = 5
NUM_COLS = 5
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 1
TRAVERSAL_MIN = -10
TRAVERSAL_MAX = 10
TRAVERSAL_RESOLUTION = 21

num_imgs = NUM_ROWS * NUM_COLS
checkpoint_dir = os.path.join(EXPERIMENT_DIR, 'checkpoints/')
log_dir = os.path.join(EXPERIMENT_DIR, 'logs/')
output_dir = os.path.join(EXPERIMENT_DIR, 'traversals/')

if not os.path.isdir(output_dir):
	os.makedirs(output_dir)

def make_svae():
	svae = SVAE(
		checkpoint_dir, log_dir,
		img_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
		num_latents=NUM_LATENTS, num_classes=10)
	return svae

def resize_data(X):
  image_list = list()
  for image in X:
    image_list.append(resize(image, output_shape=(32, 32)))
  X = np.array(image_list)
  # print(X.shape)
  # print(len(image_list))
  assert(X.shape == (len(image_list), 32, 32, 1))
  return X

svae = make_svae()

datagen = ImageDataGenerator()

mnist = input_data.read_data_sets('../data/MNIST/', reshape=False)
X_test = mnist.test.images
X_test = resize_data(X_test)
y_test = mnist.test.labels

test_datagen = datagen.flow(X_test, y_test, batch_size=num_imgs)

images, labels = next(test_datagen)

latent_means = svae.compress(images)
for latent_index in range(NUM_LATENTS):
	grids = list()
	for new_val in np.linspace(
		TRAVERSAL_MIN, TRAVERSAL_MAX, TRAVERSAL_RESOLUTION):

		latent_means_updated = latent_means.copy()
		for latent_mean_vector in latent_means_updated:
			latent_mean_vector[latent_index] = new_val

		reconstructions = svae.reconstruct_latents(latent_means_updated)
		reconstructions = reconstructions.reshape(
			NUM_ROWS, NUM_COLS, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
		reconstructions = np.column_stack(np.column_stack(reconstructions))
		grids.append(reconstructions)
	save_path = os.path.join(
		output_dir, 'traversal_{:0>2}.gif'.format(latent_index))
	imageio.mimsave(save_path, grids)