import os

# Note: run this cript only from the directory where it lives.

# Removes JPG files that are less than 3kb in size from the subdirectories (1 level down) in th
# given directory.

# path relative to the directory where this script lives
DIR = "../data/images/imagenet11/256/val"
SIZE = 3000

directory = os.path.join(os.getcwd(), DIR)

# Returns a list of full directory paths in current directory.
def get_dirs_in_curr(curr_dir):
	return [os.path.join(curr_dir, name) for name in os.listdir(curr_dir) if os.path.isdir(os.path.join(curr_dir, name))]

# Returns a list of image filenames (with full path) in the directory dir_path.
def get_image_filenames(dir_path):
	return [os.path.join(dir_path, o) for o in os.listdir(dir_path) if o.lower().endswith(".jpeg") or o.lower().endswith(".jpg")]

count = 0
dirs = get_dirs_in_curr(directory)
for directory in dirs:
	image_filenames = get_image_filenames(directory)
	for full_file in image_filenames:
		filesize = os.path.getsize(full_file)
		if filesize < SIZE:
			os.remove(full_file)
			count += 1

print "Removed " + str(count) + " files."