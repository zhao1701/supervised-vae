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

from svae import SVAE
from tensorflow.keras.preprocessing.image import ImageDataGenerator

parser = ap.ArgumentParser()
parser.add_argument(
	'--experiment_dir', type=str,
	help='Directory containing experiment resources.')
parser.add_argument(
	'--data_dir', type=str, default='../data/processed/sample/',
	help='Directory containing image data for traversals.')
parser.add_argument(
	'--num_latents', type=int, default=32,
	help='Number of latent components to traverse over.')
parser.add_argument(
	'--num_rows', type=int, default=3,
	help='Number of rows in traversal grid.')
parser.add_argument(
	'--num_cols', type=int, default=3,
	help='Number of columns in traversal grid.')
parser.add_argument(
	'--img_height', type=int, default=128,
	help='Height of image in pixels.')
parser.add_argument(
	'--img_width', type=int, default=128,
	help='Width of image in pixels.')
parser.add_argument(
	'--img_channels', type=int, default=3,
	help='Number of color channels in image.')

args = parser.parse_args()

EXPERIMENT_DIR = args.experiment_dir
DATA_DIR = args.data_dir
NUM_LATENTS = args.num_latents
NUM_ROWS = args.num_rows
NUM_COLS = args.num_cols
IMG_HEIGHT = args.img_height
IMG_WIDTH = args.img_width
IMG_CHANNELS = args.img_channels

num_imgs = NUM_ROWS * NUM_COLS
checkpoint_dir = os.path.join(EXPERIMENT_DIR, 'checkpoints/')
log_dir = os.path.join(EXPERIMENT_DIR, 'logs/')
output_dir = os.path.join(EXPERIMENT_DIR, 'traversals/')

if not os.path.isdir(output_dir):
	os.makedirs(output_dir)

datagen = ImageDataGenerator(rescale=1./255)
sample_datagen = datagen.flow_from_directory(
	DATA_DIR,
	batch_size=num_imgs,
	target_size=(128, 128),
	class_mode='binary')

images, labels = next(sample_datagen)

assert(NUM_ROWS * NUM_COLS <= len(images))

svae = SVAE(
	checkpoint_dir,
	log_dir,
	img_shape=(128, 128, 3),
	num_latents=32,
	num_classes=2)

latent_means = svae.compress(images)
for latent_index in range(NUM_LATENTS):
	grids = list()
	for new_val in np.linspace(-4, 4, 17):
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