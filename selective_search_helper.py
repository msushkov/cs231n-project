import imp

# useful stuff:
# import selective_search_helper as x
# x.write_to_file(['../harp_0000.jpg'], 'abc.csv')

SELECTIVE_SEARCH_PY = '/afs/.ir.stanford.edu/users/m/s/msushkov/selective_search_ijcv_with_python/selective_search.py'

f = imp.load_source('selective_search_ijcv_with_python', SELECTIVE_SEARCH_PY)

# SPECIFY ABSOLUTE FILE PATHS
def get_windows(image_fnames, cmd='selective_search'):
	return f.get_windows(image_fnames, cmd)

def write_to_file(image_fnames, out_file):
	windows_list = get_windows(image_fnames)
	f = open(out_file, 'w')
	f.write('filename,ymin,xmin,ymax,xmax\n')
	for i, windows in enumerate(windows_list):
		filename = image_fnames[i]
		rows = windows.shape[0]
		for j in xrange(rows):
			curr_window = windows[j]
			lst = curr_window.tolist()
			lst = [str(x) for x in lst]
			f.write(filename + ',' + ','.join(lst) + '\n')
	f.close()