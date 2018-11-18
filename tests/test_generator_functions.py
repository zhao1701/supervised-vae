import os
import sys
import pytest
import shutil

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

sys.path.append('..')

from svae import SVAE

NUM_LATENTS = 32
NUM_CLASSES = 2
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
NUM_SAMPLES = 200
CHECKPOINT_DIR = '../experiments/pytest/checkpoints/'
LOG_DIR = '../experiments/pytest/logs/'
TRAIN_DIR = '../data/processed/sample/'
VALIDATION_DIR = '../data/processed/sample/'

@pytest.fixture()
def make_generators():
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
	return train_datagen, validation_datagen

# @pytest.fixture()
def make_svae():
	svae = SVAE(
		CHECKPOINT_DIR, LOG_DIR,
		img_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
		num_latents=NUM_LATENTS, num_classes=NUM_CLASSES)
	return svae

svae = make_svae()

def reset_resources():
	directories = [CHECKPOINT_DIR, LOG_DIR]
	for directory in directories:
		if os.path.isdir(directory):
			shutil.rmtree(directory)
			os.makedirs(directory)

def test_fit_classifier_generator(make_generators):
	reset_resources()
	train_gen, val_gen = make_generators
	svae.fit_classifier_generator(train_gen, val_gen, num_epochs=2)

def test_fit_decoder_generator(make_generators):
	reset_resources()
	train_gen, val_gen = make_generators
	svae.fit_decoder_generator(train_gen, num_epochs=2)

def test_predict_generator(make_generators):
	reset_resources()
	train_gen, val_gen = make_generators
	y, y_pred = svae.predict_generator(val_gen)

def test_calc_overall_metrics(make_generators):
	reset_resources()
	train_gen, val_gen = make_generators
	acc,loss = svae._calc_classifier_metrics(train_gen)
	assert(0 <= acc <= 1)
	assert(loss > 0)