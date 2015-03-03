# Generates a single file containing image filenames and their labels.

# NOTE: must be executed from the directory from where this file lives.
# Run this on the machine where the training will take place.

import os
import os.path

directory = "data/sample"

os.chdir(directory)

out_filename = "caffe_labels_sample.txt"
cwd = os.getcwd()

def get_label_map():
	label_key_file = "label_key.txt"
	f = open(label_key_file, 'r')
	label_map = {}
	for line in f:
		vals = line.strip().split()
		word_label = vals[0]
		int_label=  vals[1]
		label_map[word_label] = int_label
	return label_map

# Returns a list of absolute directory paths in current directory.
def get_dirs_in_curr():
	return [os.path.join(cwd, name) for name in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, name))]

# Returns a list of image filenames (with full path) in the directory dir_path.
def get_image_filenames(dir_path):
	return [os.path.join(cwd, o) for o in os.listdir(dir_path) \
		if os.path.join(cwd,o).lower().endswith(".jpeg") or os.path.join(cwd,o).lower().endswith(".jpg")]


out = open(out_filename, 'w')

label_map = get_label_map()
dirs = get_dirs_in_curr()
for directory in dirs:
	curr_name = directory.split('/')[-1]
	curr_label = label_map[curr_name]

	image_filenames = get_image_filenames(directory)
	for name in image_filenames:
		out.write(name + " " + curr_label + "\n")

out.close()