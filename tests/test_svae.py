import os
import sys
import pytest
import shutil

import numpy as np

sys.path.append('..')

from svae import SVAE


NUM_SAMPLES = 100
NUM_LATENTS = 32
NUM_CLASSES = 2
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
CHECKPOINT_DIR = '../experiments/test/checkpoints/'
LOG_DIR = '../experiments/test/logs/'
# TRAVERSAL_DIR = '../experiments/test/traversal_check/'

directories = [CHECKPOINT_DIR, LOG_DIR]
for directory in directories:
	if os.path.isdir(directory):
		shutil.rmtree(directory)
		os.makedirs(directory)

@pytest.fixture()
def make_data():
	X = np.random.randn(NUM_SAMPLES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
	y = np.random.choice(np.arange(NUM_CLASSES), (NUM_SAMPLES, NUM_CLASSES))
	return X, y


@pytest.fixture()
def make_svae():
	svae = SVAE(
		CHECKPOINT_DIR, LOG_DIR,
		img_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
		num_latents=32, num_classes=2)
	return svae


def test_fit_classifier(make_svae, make_data):
	X, y = make_data
	svae = make_svae
	svae.fit_classifier(X, y, num_epochs=3)


def test_fit_decoder(make_svae, make_data):
	X, y = make_data
	svae = make_svae
	svae.fit_decoder(X, num_epochs=3)


def test_predict_proba(make_svae, make_data):
	X, y = make_data
	svae = make_svae
	predictions = svae.predict_proba(X)
	assert(predictions.shape == (NUM_SAMPLES, NUM_CLASSES))

def test_predict_labels(make_svae,make_data):
	X, y = make_data
	svae = make_svae
	probabilities = svae.predict_proba(X)
	precitions = svae.predict_label(probabilities)
	assert(precitions.shape == (NUM_SAMPLES, NUM_CLASSES))


def test_compress(make_svae, make_data):
	X, y = make_data
	svae = make_svae
	latents = svae.compress(X)
	assert(latents.shape == (NUM_SAMPLES, NUM_LATENTS))


def test_reconstruct(make_svae, make_data):
	X, y = make_data
	svae = make_svae
	reconstructions = svae.reconstruct(X)
	expected_shape = (NUM_SAMPLES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
	assert(reconstructions.shape == expected_shape)


def test_reconstruct_latents(make_svae, make_data):
	X, y = make_data
	svae = make_svae
	latents = svae.compress(X)
	reconstructions = svae.reconstruct_latents(latents)
	expected_shape = (NUM_SAMPLES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
	assert(reconstructions.shape == expected_shape)