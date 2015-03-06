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
GROUND_TRUTH_LABEL_FILE = os.environ['TEST_LABELS']
K = 2 # take the top k when computing score

caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=np.load(MEAN_FILE).mean(1).mean(1), channel_swap=(2, 1, 0), raw_scale=255)

# Load into memory the filename -> label mapping
def load_label_file():
	f = open(GROUND_TRUTH_LABEL_FILE, 'r')
	image_labels = {}
	for line in f:
		vals = line.strip().split()
		image_filename = vals[0]
		int_label = vals[1]
		image_labels[image_filename] = int(int_label)
	f.close()
	return image_labels

# mapping of filename -> ground truth label
image_labels = load_label_file()

image_filenames = image_labels.keys()
images = [caffe.io.load_image(name) for name in image_filenames]
predictions = net.predict(images)

# record errors by class label
errors = Counter()
totals = Counter()
for i, filename in enumerate(image_filenames):
	curr_predictions = predictions[i]

	idx = [i for i in xrange(len(curr_predictions))]
	preds = zip(curr_predictions, idx) # gives list of tuples (pred, index)

	sorted_predictions = sorted(preds, reverse=True)
	top_k = sorted_predictions[:K]

	top_k_labels = set([index for (pred, index) in top_k])

	true_label = image_labels[filename]

	totals[true_label] += 1
	if true_label not in top_k_labels:
		errors[true_label] += 1

print "Accuracy: "
for true_class in errors:
	accuracy = 1.0 - float(errors[true_label]) / float(totals[true_label])
	print "True label %s; accuracy = %f" % (true_label, accuracy)
