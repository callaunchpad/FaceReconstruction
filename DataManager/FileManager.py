import os
import cv2
import scipy.io as scio

def enumerate_data_paths():
    data_dir = os.getcwd() + '/300W_LP'
    for root, dirs, files in os.walk(data_dir):
        if 'Code' in root or 'landmarks' in root:
            continue
        data = set([s[:-4] for s in files])
        for file in data:
            file_prefix = root + '/' + file
            yield file_prefix

def get_datum(path):
    pic_path = path + '.jpg'
    vertex_path = path + '.mat'
    pic = cv2.imread(pic_path)
    vertices = scio.loadmat(vertex_path)['pt2d']
    return pic, vertices
