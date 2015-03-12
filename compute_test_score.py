# Computes the score for the test directory

import sys
import os

caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
from collections import Counter, defaultdict
import random

# feed chunks of this size into the cnn at a time to compute scores for all of them
CHUNK_SIZE = 200

# are we testing on the out-of-the-box AlexNet/GoogLeNet/NIN that outputs a 1000-d vector?
# (if we are, then the label indices will be messed up so need to account for that)
PREDICTING_1000_CLASSES = True

# model 2 - not finetuned CaffeNet
#PRETRAINED = "/root/cs231n-project/cnns/alexnet-11/snapshots/alexnet11_iter_4000.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/alexnet-11/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"

# model 3 - finetuned CaffeNet
#PRETRAINED = "/root/cs231n-project/cnns/cnn3/snapshots/cnn3_iter_2000.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/cnn3/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/no_augmentations/imagenet/256/imagenet11_no_aug_mean.npy"

# GoogleNet - not finetuned
PRETRAINED = "/root/cs231n-project/cnns/googlenet/reference_model/bvlc_googlenet.caffemodel"
MODEL_FILE = "/root/cs231n-project/cnns/googlenet/reference_model/deploy.prototxt"
MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"

# NIN - not finetuned
#PRETRAINED = "/root/cs231n-project/cnns/nin/reference_model/nin_imagenet.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/nin/reference_model/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"


GROUND_TRUTH_DIR = "/root/cs231n-project/data/images/val/instagram/227"

GROUND_TRUTH_LABEL_FILE = os.path.join(GROUND_TRUTH_DIR, "caffe.txt")

# alexnet predicted index -> our label index
alexnet_labels = {
	444 : 0, # bicycle
	406 : 2, # christmasstocking
	594 : 3, # harp
	283 : 5, # persiancat
	806 : 8  # soccerball
}

K = 2 # take the top k when computing score

if PREDICTING_1000_CLASSES:
	K = 20
	PRETRAINED = "/root/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
	MODEL_FILE = "/root/caffe/models/bvlc_reference_caffenet/deploy.prototxt"
	MEAN_FILE = "/root/caffe/data/ilsvrc12/imagenet_mean.npy"


TRASH = 10 # label for trash class


label_key = {
	"bicycle" : 0,
	"broccoli" : 1,
	"christmasstocking" : 2,
	"harp" : 3,
	"paintballmarker" : 4,
	"persiancat" : 5,
	"sewingmachine" : 6,
	"skateboard" : 7,
	"soccerball" : 8,
	"tennisball" : 9,
	"trash" : 10
}

key_label = {}
for key in label_key:
	key_label[label_key[key]] = key

label_indices_set = set(key_label.keys())

# Returns a list of chunks of size n, made from l.
def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]


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

# Get the class label from the filename (e.g. bike_001.jpg --> bike)
def get_class_label(filename):
	class_name = filename.split('/')[-2].split('_', 1)[0]
	return label_key[class_name]

# mapping of filename -> ground truth label
image_labels = load_label_file()

# a set of all possible correct labels
gold_labels = set(image_labels.values())
gold_labels.remove(TRASH)

# chunk the filenames of all the images we want to test
image_filenames = image_labels.keys()
filename_chunks = chunks(image_filenames, CHUNK_SIZE)

# record errors by class label
# assume working with an image whose class label number is x (trash is 10)
# fp: if prediction on image is x and fiverr label is 10
# tp: if prediction on image is x and fiverr label is x
# fn: if prediction on image is 10 and fiverr label is x
# tn: if prediction on image is 10 and fiverr label is 10

fp = Counter()
fn = Counter()
tp = Counter()
tn = Counter()

# class_label -> [(filename, class_label_index), ...]
tp_filenames = defaultdict(list)

c = 0
n = len(filename_chunks)

# image filename -> ("FP" or "FN", correct class, predicted class)
examples = {}
num_examples_to_show_fp = 10
num_examples_to_show_fn = 10

# Get the predictions for each chunk
for curr_filenames in filename_chunks:
	print "Chunk %d out of %d" % (c, n)
	c += 1

	images = [caffe.io.load_image(os.path.join(GROUND_TRUTH_DIR, name)) for name in curr_filenames]
	predictions = net.predict(images)

	for i, filename in enumerate(curr_filenames):
		curr_predictions = predictions[i]

		if PREDICTING_1000_CLASSES:
			assert len(curr_predictions) == 1000
		else:
			assert len(curr_predictions) == 11

		idx = [i for i in xrange(len(curr_predictions))]
		preds = zip(curr_predictions, idx) # gives list of tuples (pred, index)

		sorted_predictions = sorted(preds, reverse=True)
		top_k = sorted_predictions[:K]

		top_k_labels_lst = [index for (pred, index) in top_k]
		top_k_labels = set(top_k_labels_lst)

		# as far as we are concerned, AlexNet will predict one of our 5 classes, or trash
		if PREDICTING_1000_CLASSES:
			new_labels = set()
			for label in top_k_labels:
				if label in alexnet_labels:
					new_labels.add(alexnet_labels[label])
				else:
					new_labels.add(TRASH)
			top_k_labels = new_labels

		fiverr_label = image_labels[filename]
		class_label = get_class_label(filename)

		# assume working with an image whose class label number is x (trash is 10)
		# fp: if prediction on image is x and fiverr label is 10
		# tp: if prediction on image is x and fiverr label is x
		# fn: if prediction on image is 10 or y != x and fiverr label is x
		# tn: if prediction on image is 10 and fiverr label is 10

		if class_label in top_k_labels:
			if fiverr_label == TRASH:
				fp[class_label] += 1

				if num_examples_to_show_fp > 0:
					fiverr_label_word = key_label[fiverr_label]
					guesses = [key_label[label] for label in top_k_labels]
					examples[filename] = ("FP", fiverr_label_word, guesses)
					num_examples_to_show_fp -= 1
			else:
				# fiverr guy correctly labeled it as the class that it is
				assert fiverr_label == class_label, "Fiverr label = %s, class label = %s" % (fiverr_label, class_label)
				tp[class_label] += 1
				tp_filenames[class_label].append((filename, top_k_labels_lst.index(class_label)))
		elif TRASH in top_k_labels:
			if fiverr_label == class_label:
				fn[class_label] += 1

				if num_examples_to_show_fn > 0:
					fiverr_label_word = key_label[fiverr_label]
					guesses = [key_label[label] for label in top_k_labels]
					examples[filename] = ("FN", fiverr_label_word, guesses)
					num_examples_to_show_fn -= 1
			else:
				# fiverr guy correctly labeled it as trash
				assert fiverr_label == TRASH, "Fiverr label = %s" % fiverr_label
				tn[class_label] += 1
		else:
			fn[class_label] += 1


print "Scores: "
total_num = 0
for true_class in gold_labels:
	tp_val = tp[true_class]
	fp_val = fp[true_class]
	tn_val = tn[true_class]
	fn_val = fn[true_class]

	total = tp_val + fp_val + tn_val + fn_val
	total_num += total

	if total == 0:
		print "  True label %s -> # test points = %d" (key_label[true_class], total)
	else:
		accuracy = float(tp_val + tn_val) / (tp_val + fp_val + tn_val + fn_val)
		
		precision = -1.0
		if tp_val + fp_val > 0:
			precision = float(tp_val) / (tp_val + fp_val)
		
		recall = -1.0
		if tp_val + fn_val > 0:
			recall = float(tp_val) / (tp_val + fn_val)

		print "  True label %s -> # test points = %d, acc = %f; prec = %f; recall = %f" % \
			(key_label[true_class], total, accuracy, precision, recall)

print "Total # of test points = %d \n" % total_num

print examples

out_file = open('classlabel_tp.txt', 'w')
for key in key_label:
	for line in tp_filenames[key]:
		out_file.write(line + "\n")
out_file.close()


