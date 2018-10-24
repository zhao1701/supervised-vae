import os
import glob
import tarfile
import shutil

import itertools as it
import pandas as pd
import numpy as np
import argparse as ap

from scipy.io import loadmat

parser = ap.ArgumentParser()
parser.add_argument(
	'--source_dir', type=str, default='../data/raw/',
	help='Directory in which raw tar files are located.')
parser.add_argument(
	'--temp_dir', type=str, default='../data/temp/',
	help='Directory where temporary files are stored.')
parser.add_argument(
	'--destination_dir', type=str, default='../data/processed/',
	help='Directory where processed files are stored.')
parser.add_argument(
	'--train_prop', type=float, default=0.9,
	help='Proportion of data to be used for training.')
parser.add_argument(
	'--validation_prop', type=float, default=0.05,
	help='Proportion of data to be used for validation.')

args = parser.parse_args()

SOURCES = ['wiki', 'imdb']
SOURCE_DIR = args.source_dir
TEMP_DIR = args.temp_dir
DESTINATION_DIR = args.destination_dir
TRAIN_PROP = args.train_prop
VALIDATION_PROP = args.validation_prop

test_prop = 1 - TRAIN_PROP - VALIDATION_PROP

# Create directory structure for processed files
splits = ['train', 'validation', 'test']
classes = ['female', 'male']
for split, class_ in it.product(splits, classes):
	processed_dir = os.path.join(DESTINATION_DIR, split, class_)
	if not os.path.isdir(processed_dir):
		os.makedirs(processed_dir)

dfs = list()
for source in SOURCES:

	print(f'Unpacking data for {source}...')
	tar_path = os.path.join(SOURCE_DIR, f'{source}-cropped.tar')
	with tarfile.open(tar_path) as tar:
	 	tar.extractall(TEMP_DIR)

	print(f'Reading metadata for {source}...')
	mat_path = os.path.join(
		TEMP_DIR,
		f'{source}_crop',
		f'{source}.mat')
	mat = loadmat(mat_path)
	mat = mat[source][0][0]
	data = dict(
		filename=mat[2].squeeze(),
		gender=mat[3].squeeze(),
		name=mat[4].squeeze(),
		img_year=mat[1].squeeze())

	df = pd.DataFrame(data)
	df['filename'] = df['filename'].apply(lambda x: x[0])
	df['name'] = df['name'].apply(lambda x: x[0] if x.size == 1 else '')
	dfs.append(df)

	print(f'Processing files for {source}...')
	for gender, indicator in zip(['female', 'male'], [0, 1]):

		# Get filenames for gender
		df_gender = df[df['gender'] == indicator]
		df_gender_filenames = df_gender['filename'].apply(
			lambda x: os.path.join(TEMP_DIR, f'{source}_crop', x))
		df_gender_filenames = df_gender_filenames.values

		# Shuffle
		num_files = len(df_gender_filenames)
		shuffled_indices = np.random.permutation(range(num_files))
		df_gender_filenames = df_gender_filenames[shuffled_indices]

		# Partition - each value is a boundary index between splits
		train_val = int(num_files * TRAIN_PROP)
		val_test = int(num_files * (TRAIN_PROP + VALIDATION_PROP))

		print(f'Processing training data for {gender} class...')
		df_gender_filenames_train = df_gender_filenames[:train_val]
		train_gender_dest = os.path.join(DESTINATION_DIR, 'train', gender)
		for file in df_gender_filenames_train:
			shutil.copy2(file, train_gender_dest)

		print(f'Processing validation data for {gender} class...')
		df_gender_filenames_val = df_gender_filenames[train_val:val_test]
		val_gender_dest = os.path.join(DESTINATION_DIR, 'validation', gender)
		for file in df_gender_filenames_val:
			shutil.copy2(file, val_gender_dest)

		print(f'Processing test data for {gender} class...')
		df_gender_filenames_test = df_gender_filenames[val_test:]
		test_gender_dest = os.path.join(DESTINATION_DIR, 'test', gender)
		for file in df_gender_filenames_test:
			shutil.copy2(file, test_gender_dest)

print('Writing processed metadata CSV.')
df_both = pd.concat(dfs, axis='rows')
df_both['filename'] = df_both['filename'].apply(lambda x: x[3:])
df_both.to_csv(os.path.join(DESTINATION_DIR, 'metadata.csv'))

print('Cleaning up temp directory...')
os.system('rm -r ../data/temp/*')