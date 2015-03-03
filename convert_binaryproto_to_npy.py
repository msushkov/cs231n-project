import sys
import os

caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

# Converts a file in binaryproto format to a file in nby format. (Called by make_mean.sh)

if len(sys.argv) != 3:
	print "Usage: python convert_binaryproto_to_npy.py file.binaryproto out.npy"
sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(sys.argv[1], 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
out = arr[0]
np.save( sys.argv[2], out)