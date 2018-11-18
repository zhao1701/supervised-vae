import os
import sys
import csv
import pytest
import shutil

import numpy as np
import argparse as ap

sys.path.append('..')

from svae import SVAE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, roc_auc_score

parser = ap.ArgumentParser()
parser.add_argument(
	'--experiment_dir', type=str,
	help='Directory in which logs, checkpoints, and other resources are saved.')
parser.add_argument(
	'--decoder', action='store_true',
	help='Whether to train the decoder instead of classifier network.')
parser.add_argument(
	'--test', action='store_true',
	help='Loads and evaluates the SVAE on the test set.')
parser.add_argument(
	'--learning_rate', type=float, default=1e-4,
	help='Learning rate during model training.')
parser.add_argument(
	'--batch_size', type=int, default=128,
	help='Number of samples per batch.')
parser.add_argument(
	'--num_epochs', type=int, default=20,
	help='Number of epochs.')
parser.add_argument(
	'--beta', type=float, default=1,
	help='Amount by which to weight latent loss in classifier loss.')
parser.add_argument(
	'--train_dir', type=str, default='../data/processed/train/',
	help='Directory containing the training data.')
parser.add_argument(
	'--validation_dir', type=str, default='../data/processed/validation/',
	help='Directory containing the validation data.')
parser.add_argument(
	'--test_dir', type=str, default='../data/processed/test/',
	help='Directory containing the test data.')
parser.add_argument(
	'--num_latents', type=int, default=32,
	help='Number of latent components in variational layer.')
parser.add_argument(
	'--img_height', type=int, default=128,
	help='Height of the image.')
parser.add_argument(
	'--img_width', type=int, default=128,
	help='Width of the image.')
parser.add_argument(
	'--img_channels', type=int, default=3,
	help='Number of color channels in the image.')

args = parser.parse_args()

NUM_LATENTS = args.num_latents
IMG_HEIGHT = args.img_height
IMG_WIDTH = args.img_width
IMG_CHANNELS = args.img_channels

EXPERIMENT_DIR = args.experiment_dir
TRAIN_DIR = args.train_dir
VALIDATION_DIR = args.validation_dir
TEST_DIR = args.test_dir

DECODER = args.decoder
TEST = args.test
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
BETA = args.beta

checkpoint_dir = os.path.join(EXPERIMENT_DIR, 'checkpoints')
log_dir = os.path.join(EXPERIMENT_DIR, 'logs')

directories = [checkpoint_dir, log_dir]
for directory in directories:
	if os.path.isdir(directory):
		shutil.rmtree(directory)
		os.makedirs(directory)

def make_svae():
	svae = SVAE(
		checkpoint_dir, log_dir,
		img_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
		num_latents=NUM_LATENTS, num_classes=2)
	return svae

datagen = ImageDataGenerator(rescale=1./255)

train_datagen = datagen.flow_from_directory(
	TRAIN_DIR,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	batch_size=BATCH_SIZE,
	class_mode='binary')

validation_datagen = datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	batch_size=BATCH_SIZE,
	class_mode='binary')

test_datagen = datagen.flow_from_directory(
	TEST_DIR,
	target_size=(IMG_HEIGHT, IMG_WIDTH),
	batch_size=BATCH_SIZE,
	class_mode='binary')

svae = make_svae()
if not TEST:
	print('Logging hyperparmeters...')
	log_path = os.path.join(EXPERIMENT_DIR, 'hyperparameters.txt')
	with open(log_path, 'at') as f:
		lines = list()
		if not DECODER:
			lines.append('CLASSIFIER')
		else:
			lines.append('DECODER')
		lines.append('=' * 10)
		lines.append('num_epochs: {}\n'.format(NUM_EPOCHS))
		lines.append('batch_size: {}\n'.format(BATCH_SIZE))
		lines.append('learning_rate: {}\n'.format(LEARNING_RATE))
		lines.append('beta: {}\n'.format(BETA))
		lines.append('num_latents: {}\n'.format(BETA))
		lines.append('\n')
		f.writelines(lines)

	if not DECODER:
		print('Training classifier...')
		svae.fit_classifier_generator(
			train_datagen,
			validation_datagen, 
			num_epochs=NUM_EPOCHS,
			learning_rate=LEARNING_RATE,
			beta=BETA)
	else:
		print('Training decoder...')
		svae.fit_decoder_generator(
			train_datagen,
			validation_datagen,
			num_epochs=NUM_EPOCHS,
			learning=LEARNING_RATE)
else:
	print('Evaluating SVAE...')
	accuracy, classifier_loss = svae._calc_classifier_metrics(test_datagen)
	decoder_loss = svae._calc_decoder_metrics(test_datagen)

	y_true, y_proba = svae.predict_generator(test_datagen, labels=False)
	y_true = y_true[:,1]
	y_proba = y_proba[:,1]
	y_pred = y_proba >= 0.5

	precision = precision_score(y_true, y_pred)
	recall = recall_score(y_true, y_pred)
	auroc = roc_auc_score(y_true, y_proba)

	csv_path = os.path.join(EXPERIMENT_DIR, 'test_results.csv')
	print('Writing results to {}'.format(csv_path))
	with open(csv_path, 'w') as f:
		writer = csv.writer(f)
		writer.writerow([
			'accuracy',
			'precision',
			'recall',
			'auroc',
			'classifier_loss',
			'decoder_loss'])
		writer.writerow([
			accuracy,
			precision,
			recall,
			auroc,
			classifier_loss,
			decoder_loss])