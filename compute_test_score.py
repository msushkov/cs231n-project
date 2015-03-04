# Computes the score for the test directory

import sys
import os

caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

MODEL_FILE = os.environ['MODEL_FILE']
PRETRAINED = os.environ['PRETRAINED_NETWORK']
MEAN_FILE = os.environ['MEAN_FILE_NPY']
GROUND_TRUTH_LABEL_FILE = os.environ['TEST_LABELS']

caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=np.load(MEAN_FILE).mean(1).mean(1), channel_swap=(2, 1, 0), raw_scale=255)

# Load into memory the filename -> label mapping
def load_label_file():
	f = open(GROUND_TRUTH_LABEL_FILE, 'r')
	image_labels = {}
	for line in f:
		vals = line.strip().split()
		image_filename = vals[0].split('/')[-1]
		int_label = vals[1]
		image_labels[image_filename] = int(int_label)
	f.close()
	return image_labels

# mapping of filename -> ground truth label
image_labels = load_label_file()

image_filenames = image_labels.keys()
predictions = net.predict([image_filenames])

errors = {} # filename -> (true label, prediction)
num_errors = 0
for i, filename in enumerate(image_filenames):
	curr_predictions = predictions[i]
	best_prediction = curr_predictions.argmax()
	true_label = image_labels[filename]
	if true_label != best_prediction:
		num_errors += 1
		errors[filename] = (true_label, best_prediction)

num_test = len(image_labels)
accuracy = 1.0 - float(num_errors) / float(num_test)

print "Number of errors = %d, total # test = %d" % (num_errors, num_test)
print "Accuracy = %f" % accuracy
print "\nErrors:"
print errors
