# Computes the score for the test directory

import sys
import os

caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
from collections import Counter

MODEL_FILE = os.environ['MODEL_FILE']
PRETRAINED = os.environ['PRETRAINED_NETWORK']
MEAN_FILE = os.environ['MEAN_FILE_NPY']
TEST_DATA_ROOT = os.environ['TEST_DATA_ROOT']
SYNSET_WORDS = os.environ['SYNSET_WORDS']

image_filenames = [
	os.path.join(TEST_DATA_ROOT, "harp/harp_0001.jpg")
]

def load_synset_words():
	f = open(SYNSET_WORDS, 'r')
	classes = {}
	for i, line in enumerate(f):
		synset_name = line.strip().split(' ', 1)[-1]
		classes[i] = synset_name
	f.close()
	return classes


synsets = load_synset_words()


caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=np.load(MEAN_FILE).mean(1).mean(1), channel_swap=(2, 1, 0), raw_scale=255)

images = [caffe.io.load_image(name) for name in image_filenames]
predictions = net.predict(images)

for i in range(len(predictions)):
	curr = predictions[i]
	preds = curr.tolist()
	idx = [i for i in xrange(len(preds))]
	preds = zip(preds, idx)
	preds.sort(reverse=True)

	result = [(prob, synsets[idx]) for (prob, idx) in preds]
	print result[:20]


