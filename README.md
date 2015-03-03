# cs231n-project

- To run caffe on a new dataset:
	- put the JPEG's into a train/ and val/ directories (use split_train_val.py)
	- generate a label file for caffe using generate_caffe_path_label.py (on each line: filepath label)
	- run create_leveldb.sh (modified version of caffe's create_imagenet.sh) with the appropriate paths for the train and val directories
	- if we wish to use the imagenet image means during training, make sure that file has been downloaded (follow caffe instructions)
	- if we wish to use the current dataset's image means during training, run make_mean.sh (modified version of caffe's make_imagenet_mean.sh) 
	- follow the remaining instructions at https://github.com/BVLC/caffe/issues/550