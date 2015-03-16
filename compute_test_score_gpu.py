# Computes the score for the test directory

import sys
import os
import random

caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
from collections import Counter, defaultdict
import random

# feed chunks of this size into the cnn at a time to compute scores for all of them
CHUNK_SIZE = 200

TRASH = 10

# are we testing on the out-of-the-box AlexNet/GoogLeNet/NIN that outputs a 1000-d vector?
# (if we are, then the label indices will be messed up so need to account for that)
PREDICTING_1000_CLASSES = True

K = 20 # take the top k when computing score

#if PREDICTING_1000_CLASSES:
#	K = 20


# model 3 - cnn3
#PRETRAINED = "/root/cs231n-project/cnns/cnn3/snapshots/cnn3_iter_2000.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/cnn3/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/no_augmentations/imagenet/256/imagenet11_no_aug_mean.npy"


# baseline_cnn
#PRETRAINED = "/root/cs231n-project/cnns/baseline_cnn/snapshots/baseline_cnn_iter_.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/baseline_cnn/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/no_augmentations/imagenet/256/imagenet11_no_aug_mean.npy"


# CaffeNet - not finetuned
PRETRAINED = "/root/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
MODEL_FILE = "/root/caffe/models/bvlc_reference_caffenet/deploy.prototxt"
MEAN_FILE = "/root/caffe/data/ilsvrc12/imagenet_mean.npy"

# CaffeNet - finetuned (only last layer trained)
#PRETRAINED = "/root/cs231n-project/cnns/alexnet-11/fix_all_other_layers/snapshots/alexnet-11_fix_all_other_layers_iter_1000.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/alexnet-11/fix_all_other_layers/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"

# CaffeNet - finetuned (all layers trained)
#PRETRAINED = "/root/cs231n-project/cnns/googlenet/train_all_layers/snapshots/alexnet-11_train_all_layers_iter_1000.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/googlenet/train_all_layers/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"



# GoogleNet - not finetuned
#PRETRAINED = "/root/cs231n-project/cnns/googlenet/reference_model/bvlc_googlenet.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/googlenet/reference_model/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"

# GoogleNet - finetuned (only last layer trained)
#PRETRAINED = "/root/cs231n-project/cnns/googlenet/fix_all_other_layers/snapshots/googlenet11_fix_all_other_layers_iter_2000.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/googlenet/fix_all_other_layers/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"

# GoogleNet - finetuned (all layers trained)
#PRETRAINED = "/root/cs231n-project/cnns/googlenet/train_all_layers/snapshots/googlenet11_train_all_layers_iter_2000.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/googlenet/train_all_layers/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"



# NIN - not finetuned
#PRETRAINED = "/root/cs231n-project/cnns/nin/reference_model/nin_imagenet.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/nin/reference_model/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"

# NIN - finetuned (only last layer trained)
#PRETRAINED = "/root/cs231n-project/cnns/nin/fix_all_other_layers/nin11_fix_all_other_layers_iter_1000.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/nin/fix_all_other_layers/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"

# NIN - finetuned (all layers trained)
#PRETRAINED = "/root/cs231n-project/cnns/nin/train_all_layers/nin11_train_all_layers_iter_1000.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/nin/train_all_layers/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"


# Googlenet data experiments

# no_aug_IM_IG
#PRETRAINED = "/root/cs231n-project/cnns/googlenet/train_all_layers/data_experiments/snapshots/no_aug_IM_IG_iter_.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/googlenet/train_all_layers/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"

# aug_1_IM
#PRETRAINED = "/root/cs231n-project/cnns/googlenet/train_all_layers/data_experiments/snapshots/googlenet11_train_all_layers_iter_.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/googlenet/train_all_layers/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"

# aug_1_IM_IG
#PRETRAINED = "/root/cs231n-project/cnns/googlenet/train_all_layers/data_experiments/snapshots/googlenet11_train_all_layers_iter_.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/googlenet/train_all_layers/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"

# aug_2_IM
#PRETRAINED = "/root/cs231n-project/cnns/googlenet/train_all_layers/data_experiments/snapshots/googlenet11_train_all_layers_iter_.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/googlenet/train_all_layers/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"

# aug_2_IM_IG
#PRETRAINED = "/root/cs231n-project/cnns/googlenet/train_all_layers/data_experiments/snapshots/googlenet11_train_all_layers_iter_.caffemodel"
#MODEL_FILE = "/root/cs231n-project/cnns/googlenet/train_all_layers/deploy.prototxt"
#MEAN_FILE = "/root/cs231n-project/data/image_means/ilsvrc12/imagenet_mean.npy"


# the val/test data
#GROUND_TRUTH_DIR = "/root/cs231n-project/data/images/val/instagram/227"
GROUND_TRUTH_DIR = "/root/cs231n-project/test_data/instagram"

GROUND_TRUTH_LABEL_FILE = os.path.join(GROUND_TRUTH_DIR, "caffe.txt")

# alexnet predicted index -> our label index
alexnet_labels = {
	444 : 0, # bicycle
	406 : 2, # christmasstocking
	594 : 3, # harp
	283 : 5, # persiancat
	806 : 8, # soccerball
	852 : 9, # tennisball
	786 : 6, # sewingmachine
	937 : 1  # broccoli
}

alexnet_label_set = set(alexnet_labels.values())
alexnet_label_set.add(TRASH)

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
	c = 0
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

# mapping of ground truth label -> list of filenames
label_to_filenames = {}
for key in image_labels:
	label = image_labels[key]
	if label not in label_to_filenames:
		label_to_filenames[label] = []
	label_to_filenames[label].append(key)

# a set of all possible correct labels
gold_labels = set(image_labels.values())

if PREDICTING_1000_CLASSES:
	# remove from gold_labels the filenames that arent in alexnet's 1000-class output
	for x in key_label:
		if x not in alexnet_label_set:
			gold_labels.remove(x)

assert TRASH in gold_labels

#gold_labels = set([0])

# limit each ground truth label to have 300 filenames
num_test_files_per_cat = 300
image_filenames = []
#random.seed(10) # the shuffle will always give the same result. that's what we want
for key in gold_labels:
	fnames = label_to_filenames[key]
	#random.shuffle(fnames)
	fnames_limited = fnames[:num_test_files_per_cat]
	print len(fnames_limited)
	for fname in fnames_limited:
		image_filenames.append(fname)

image_filenames = image_filenames[:10]
image_filenames.append('bicycle/bicycle_i_0629_0.jpg')
print len(image_filenames)

# chunk the filenames of all the images we want to test
#image_filenames = image_labels.keys()
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

		# as far as we are concerned, AlexNet will predict one of our 5 classes, or trash
		if PREDICTING_1000_CLASSES:
			new_labels = set()
			for label in top_k_labels_lst:
				if label in alexnet_labels:
					new_labels.add(alexnet_labels[label])
				else:
					new_labels.add(TRASH)
			top_k_labels_lst = list(new_labels)

		top_k_labels = set(top_k_labels_lst)

		fiverr_label = image_labels[filename]
		class_label = get_class_label(filename)

		# assume working with an image whose class label number is x (trash is 10)
		# fp: if prediction on image is x and fiverr label is 10
		# tp: if prediction on image is x and fiverr label is x
		# fn: if prediction on image is 10 or y != x and fiverr label is x
		# tn: if prediction on image is 10 and fiverr label is 10

		#print filename, class_label, fiverr_label, top_k_labels

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

				index_of_class_label = top_k_labels_lst.index(class_label)
				tp_filenames[class_label].append((filename, index_of_class_label))
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
total_acc = 0.0
total_prec = 0.0
total_recall = 0.0
for true_class in gold_labels:
	if true_class == TRASH:
		continue

	tp_val = tp[true_class]
	fp_val = fp[true_class]
	tn_val = tn[true_class]
	fn_val = fn[true_class]

	total = tp_val + fp_val + tn_val + fn_val
	total_num += total
	total_acc += (tp_val + tn_val)

	if total == 0:
		print "  True label %s -> # test points = %d" % (key_label[true_class], total)
	else:
		accuracy = float(tp_val + tn_val) / (tp_val + fp_val + tn_val + fn_val)
		
		precision = -1.0
		if tp_val + fp_val > 0:
			precision = float(tp_val) / (tp_val + fp_val)

		if precision == -1:
			precision = 0
		
		recall = -1.0
		if tp_val + fn_val > 0:
			recall = float(tp_val) / (tp_val + fn_val)

		if recall == -1:
			recall = 0

		total_prec += precision
		total_recall += recall

		print "  True label %s -> # test points = %d; TP = %d; TN = %d; FP = %d; FN = %d; acc = %f; prec = %f; recall = %f" % \
			(key_label[true_class], total, tp_val, tn_val, fp_val, fn_val, accuracy, precision, recall)

print "Total # of test points = %d \n" % total_num
print "Average accuracy = " + str(total_acc / float(total_num)) + "\n"
print "Average precision = " + str(total_prec / float(len(gold_labels))) + "\n"
print "Average recall = " + str(total_recall / float(len(gold_labels))) + "\n"

print examples

out_file = open('classlabel_tp.txt', 'w')
for key in key_label:
	for line in tp_filenames[key]:
		out_file.write(str(line) + "\n")
out_file.close()


