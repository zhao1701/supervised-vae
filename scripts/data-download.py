import os
import glob
import requests
import tarfile

WIKI_URL = (
	'https://data.vision.ee.ethz.ch/cvl/rrothe/'
	'imdb-wiki/static/wiki_crop.tar')
IMDB_URL = (
	'https://data.vision.ee.ethz.ch/cvl/rrothe/'
	'imdb-wiki/static/imdb_crop.tar')

DESTINATION_DIR = '../data/raw/'
wiki_path = os.path.join(DESTINATION_DIR, 'wiki-cropped.tar')
imdb_path = os.path.join(DESTINATION_DIR, 'imdb-cropped.tar')

if os.path.isfile(wiki_path):
	print('WIKI tarfile already exists.')
else:
	print('Downloading WIKI images...')
	response = requests.get(WIKI_URL)
	content = response.content
	wiki_path = os.path.join(DESTINATION_DIR, 'wiki-cropped.tar')
	with open(wiki_path, 'wb') as f:
		f.write(content)

if os.path.isfile(imdb_path):
	print('IMDB tarfile already exists.')
else:
	print('Downloading IMDB images...')
	response = requests.get(IMDB_URL)
	content = response.content
	with open(imdb_path, 'wb') as f:
		f.write(content)