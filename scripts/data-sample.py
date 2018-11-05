import os
import glob
import numpy
import shutil

import numpy as np
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument(
	'-n', '--num_samples', type=int, default=100,
	help='the number of training examples to copy to the sample directory')
parser.add_argument(
	'-t', '--train_dir', type=str, default='../data/processed/train/',
	help='the directory in which the training data is located')
args = parser.parse_args()

NUM_SAMPLES = args.num_samples
TRAIN_DIR = args.train_dir

sample_dir = os.path.join(
	'/'.join(TRAIN_DIR.split('/')[:-2]), 'sample/')
labels = os.listdir(TRAIN_DIR)
num_labels = len(labels)
num_samples_per_label = NUM_SAMPLES // num_labels
print(f'Found {num_samples_per_label} labels: {labels}')

print('Creating samples...')
for label in labels:
	sample_label_dir = os.path.join(sample_dir, label)
	if not os.path.isdir(sample_label_dir):
		os.makedirs(sample_label_dir)

	glob_path = os.path.join(TRAIN_DIR, label, '*')
	image_paths = glob.glob(glob_path)
	sampled_image_paths = np.random.choice(image_paths, num_samples_per_label)

	for sampled_image_path in sampled_image_paths:
		shutil.copy2(sampled_image_path, sample_label_dir)

