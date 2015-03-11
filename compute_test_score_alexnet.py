# Computes the score for the test directory

# Runs on out of the box AlexNet.

import sys
import os

caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
from collections import Counter

PRETRAINED = "/root/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
MODEL_FILE = "/root/caffe/models/bvlc_reference_caffenet/deploy.prototxt"
MEAN_FILE = "/root/caffe/data/ilsvrc12/imagenet_mean.npy"
GROUND_TRUTH_LABEL_FILE = "/root/cs231n-project/data/images/val/instagram/227/caffe.txt"

K = 1 # take the top k when computing score

TRASH = 10 # label for trash class

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
# assume working with an image whose true label number is x (trash is 10)
# fp: if prediction on image is x and true label is 10
# tp: if prediction on image is x and true label is x
# fn: if prediction on image is 10 and true label is x
# tn: if prediction on image is 10 and true label is 10 

fp = {}
fn = {}
tp = {}
tn = {}

for i, filename in enumerate(image_filenames):
	curr_predictions = predictions[i]

	idx = [i for i in xrange(len(curr_predictions))]
	preds = zip(curr_predictions, idx) # gives list of tuples (pred, index)

	sorted_predictions = sorted(preds, reverse=True)
	top_k = sorted_predictions[:K]

	top_k_labels = set([index for (pred, index) in top_k])

	true_label = image_labels[filename]

	# if cnn predicted trash as one of the top labels
	if TRASH in tok_k_labels:
		if true_label == TRASH:
			tn[true_label] += 1
		else:
			fn[true_label] += 1
	else:
		if true_label in top_k_labels:
			tp[true_label] += 1
		else:
			fp[true_label] += 1

print "Scores: "
for true_class in tp:
	tp = tp[true_class]
	fp = fp[true_class]
	tn = tn[true_class]
	fn = fn[true_class]

	accuracy = float(tp + tn) / (tp + fp + tn + fn)
	precision = float(tp) / (tp + fp)
	recall = float(tp) / (tp + fn)

	print "  True label %s: accuracy = %f; precision = %f; recall = %f" % (true_label, accuracy, precision, recall)
