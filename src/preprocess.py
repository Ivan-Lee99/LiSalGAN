import pickle
import numpy as np
import skimage.io as io
import scipy

file = pickle.load(open(r'../vgg16.pkl', 'rb'), encoding='bytes')
k = [key for key, value in file.items()]
print(k)


def preprocess_fixmaps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), 1, shape_r, shape_c))

    for i, path in enumerate(paths):
        gazes = scipy.io.loadmat(path)["gaze"]
        coords = []
        for gaze in gazes:
            coords.extend(gaze[0][2])
        for coord in coords:
            if coord[1] >= 0 and coord[1] < shape_r and coord[0] >= 0 and coord[0] < shape_c:
                ims[i, 0, coord[1], coord[0]] = 1.0

    return ims