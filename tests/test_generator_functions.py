import os
import sys
import pytest

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
CHECKPOINT_DIR = '../experiments/test/checkpoints/'
LOG_DIR = '../experiments/test/logs/'
TRAIN_DIR = '../data/processed/sample/'
VALIDATION_DIR = '../data/processed/validation/'

@pytest.fixture()
def make_generators():
	datagen = ImageDataGenerator()

	X = np.random.randn(NUM_SAMPLES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
	y = np.random.choice(np.arange(NUM_CLASSES), NUM_SAMPLES)

	train_datagen = datagen.flow(X, y, batch_size=32)
	validation_datagen = datagen.flow(X, y, batch_size=32)
	return train_datagen, validation_datagen

@pytest.fixture()
def make_svae():
	svae = SVAE(
		CHECKPOINT_DIR, LOG_DIR,
		img_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
		num_latents=NUM_LATENTS, num_classes=NUM_CLASSES)
	return svae


def test_fit_classifier_generator(make_svae, make_generators):
	train_gen, val_gen = make_generators
	svae = make_svae
	svae.fit_classifier_generator(train_gen, val_gen, num_epochs=2)


def test_fit_decoder_generator(make_svae, make_generators):
	train_gen, val_gen = make_generators
	svae = make_svae
	svae.fit_decoder_generator(train_gen, num_epochs=2)


def test_predict_generator(make_svae, make_generators):
	train_gen, val_gen = make_generators
	svae = make_svae
	y, y_pred = svae.predict_generator(val_gen)
	assert(y.shape == (NUM_SAMPLES, NUM_CLASSES))
	assert(y_pred.shape == (NUM_SAMPLES, NUM_CLASSES))


def test_calc_overall_metrics(make_svae, make_generators):
	train_gen, val_gen = make_generators
	svae = make_svae
	acc,loss = svae._calc_overall_metrics(train_gen)
	assert(0 <= acc <= 1)
	assert(loss > 0)


