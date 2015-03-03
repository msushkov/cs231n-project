# NOTE: run this script from the directory where it lives.

# Given a directory that contains subfolders of images of a given class (data/sample/{0,1,2,...}),
# creates train and val directories data/sample/train/{0,1,2,...} and data/sample/val/{0,1,2,...}
# where the images in each class folder have been split (e.g. data/sample/train/0 contains 70% of
# the data/sample/0 images, and data/sample/val/0 contains 30% of the data/sample/0 images).

import os
from random import shuffle

###
# USER-DEFINED PARAMS
###

train_split = 0.7

DIR = "data/sample" # train and val will be in here
TEMP_DIR = "data/temp"

##################

# full paths to directories
dirr = os.path.join(os.getcwd(), DIR)
temp_dir = os.path.join(os.getcwd(), TEMP_DIR)

# which directory do we want to split into train and val?
train_dir = os.path.join(dirr, "train")
val_dir = os.path.join(dirr, "val")

# create the temp dir and put all the data there
os.mkdir(os.path.join(os.getcwd(), temp_dir))
os.system("mv %s %s" % (os.path.join(dirr, "*"), temp_dir))

# remove everything from the original directory since everything is in temp now
os.chdir(dirr)
os.system("rm -rf *")

# create the train and val directories
os.mkdir(train_dir)
os.mkdir(val_dir)

# copy temp into both train and val directories
os.system("cp -r %s %s" % (os.path.join(temp_dir, "*"), train_dir))
os.system("cp -r %s %s" % (os.path.join(temp_dir, "*"), val_dir))

# delete 30% of jpg files from each subfolder in train directory;
# delete the other 70% of jpg files from each subfolder in val directory

os.chdir(temp_dir)

# Returns a list of directory names in current directory. curr_dir is the full path.
def get_dirs_in_curr(curr_dir):
	return [name for name in os.listdir(curr_dir) if os.path.isdir(os.path.join(curr_dir, name))]

# Returns a list of image filenames in the directory dir_path. dir_path is the full path.
def get_image_filenames(dir_path):
	return [o for o in os.listdir(dir_path) if o.lower().endswith(".jpeg") or o.lower().endswith(".jpg")]

# directory is the full path to the directory; filenames is a set of filenames to remove
def remove_filenames(directory, filenames):
	for fname in filenames:
		full_path = os.path.join(directory, fname)
		if os.path.exists(full_path):
			os.remove(full_path)

# directory name -> {train/val -> [set of filenames]}
train_or_val = {}

directories = get_dirs_in_curr(os.getcwd())
for directory in directories:
	train_or_val[directory] = {}
	filenames = get_image_filenames(directory)
	shuffle(filenames)
	
	train_end_index = int(len(filenames) * train_split)
	train_filenames = set(filenames[:train_end_index])
	val_filenames = set(filenames[train_end_index:])

	train_or_val[directory]["train"] = train_filenames
	train_or_val[directory]["val"] = val_filenames

# remove the val files from train
print "Removing val files from train..."
directories = get_dirs_in_curr(train_dir)
for directory in directories:
	full = os.path.join(train_dir, directory)
	remove_filenames(full, train_or_val[directory]["val"])

# remove the train files from val
print "Removing train files from val..."
directories = get_dirs_in_curr(val_dir)
for directory in directories:
	full = os.path.join(val_dir, directory)
	remove_filenames(full, train_or_val[directory]["train"])

# remove the temp directory
os.chdir(temp_dir)
os.chdir("..")
os.system("rm -rf %s" % temp_dir)
