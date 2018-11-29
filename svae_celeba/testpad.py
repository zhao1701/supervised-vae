import os
import sys
import pytest
import shutil

import numpy as np

sys.path.append('..')

from svae import SVAE
from tensorflow.keras.preprocessing.image import ImageDataGenerator

NUM_LATENTS = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
CHECKPOINT_DIR = '../experiments/test/checkpoints/'
LOG_DIR = '../experiments/test/logs/'
TRAIN_DIR = '../data/processed/sample/'
VALIDATION_DIR = '../data/processed/sample/'

# directories = [CHECKPOINT_DIR, LOG_DIR]
# for directory in directories:
# 	if os.path.isdir(directory):
# 		shutil.rmtree(directory)
# 		os.makedirs(directory)

def make_svae():
	svae = SVAE(
		CHECKPOINT_DIR, LOG_DIR,
		img_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
		num_latents=32, num_classes=2)
	return svae

def make_data():
	X = np.random.randn(NUM_SAMPLES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
	y = np.random.choice(np.arange(NUM_CLASSES), (NUM_SAMPLES, NUM_CLASSES))
	return X, y

datagen = ImageDataGenerator(rescale=1./255)

train_datagen = datagen.flow_from_directory(
	TRAIN_DIR,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	batch_size=10,
	class_mode='binary')

validation_datagen = datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	batch_size=10,
	class_mode='binary')

svae = make_svae()
# svae.fit_classifier_generator(
# 	train_datagen, validation_datagen,
# 	num_epochs=20)

svae.fit_decoder_generator(
	train_datagen, validation_datagen,
	num_epochs=20)
