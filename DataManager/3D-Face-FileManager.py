import os
import cv2
import scipy.io as scio
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Generator that yields size 2 tuples containing the paths of matching images and 3D face vertices


def enumerate_data_paths():
    data_dir = str(Path(os.path.realpath(__file__)
                        ).parent.parent) + "\\300W-3D-Face"
    for root, dirs, files in os.walk(data_dir):
        data = set([s[:-4] for s in files])
        for file in data:
            root_path = str(Path(root).parent.parent)
            pic_path = root_path + \
                root[len(root_path):] + '\\' + file + "_0.jpg"
            vertex_path = root + '\\' + file + ".mat"
            yield pic_path, vertex_path

# Given a tuple of the paths of matching images and 3D face vertices returns the image and vertices in [[x...][y...][z...]] form


def get_datum(paths):
    pic_path = paths[0]
    vertex_path = paths[1]
    pic = cv2.imread(pic_path)
    vertices = scio.loadmat(vertex_path)['Fitted_Face']
    return pic, vertices

# plots vertices


def plot_verticies(vertices):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    xdata = vertices[0]
    ydata = vertices[1]
    zdata = vertices[2]
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    plt.show()


# print([min(get_datum(paths)[1][0]) for paths in list(enumerate_data_paths())[:10]])
# plot_verticies(get_datum(next(enumerate_data_paths()))[1])
