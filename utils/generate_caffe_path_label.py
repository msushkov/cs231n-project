# Generates a single file containing image filenames and their labels.

# NOTE: must be executed from the directory from where this file lives.
# Run this on the machine where the training will take place.

import os
import os.path

###
# USER_DEFINED VARIABLES
###

# for which directory do we want to generate the caffe.txt file?
DIR = "../data/sample/gold"

###############

label_key_file = os.environ['LABEL_KEY']
out_filename = os.environ['LABEL_FILENAME']

directory = os.path.join(os.getcwd(), DIR)
out_file_full_path = os.path.join(directory, out_filename)
label_key_full_path = os.path.join(directory, label_key_file)

def get_label_map():
	label_map = {}
	f = open(label_key_full_path, 'r')
	for line in f:
		vals = line.strip().split()
		word_label = vals[0]
		int_label=  vals[1]
		label_map[word_label] = int_label
	f.close()
	return label_map

# Returns a list of full directory paths in current directory.
def get_dirs_in_curr(curr_dir):
	return [os.path.join(curr_dir, name) for name in os.listdir(curr_dir) if os.path.isdir(os.path.join(curr_dir, name))]

# Returns a list of image filenames (with full path) in the directory dir_path.
def get_image_filenames(dir_path):
	return [os.path.join(dir_path, o) for o in os.listdir(dir_path) if o.lower().endswith(".jpeg") or o.lower().endswith(".jpg")]


out = open(out_file_full_path, 'w')
label_map = get_label_map()
dirs = get_dirs_in_curr(directory)

for directory in dirs:
	curr_name = directory.split('/')[-1]
	curr_label = label_map[curr_name]

	image_filenames = get_image_filenames(directory)
	for name in image_filenames:
		# record only classname/filename.jpg, not the full path
		vals = name.split('/')
		fname = os.path.join(vals[-2], vals[-1])
		out.write(fname + " " + curr_label + "\n")

out.close()