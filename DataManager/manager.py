# Program to convert to the numpy matrix into voxels - 3D window with
import numpy as np
import scipy
import scipy.io as scio
import os
from PIL import Image

max_z = 248

AFW_size = 313
HELEN_size = 2253
IBUG_size = 105
LFPW_size = 996

# Image paths
data_path = './300W-3D/'
subsets = ['AFW', 'HELEN', 'IBUG', 'LFPW']

data_path = './preprocessed/'

def convert_to_voxels(vertices):
    if vertices.size:

        # Scale z-coordinates to fit in [0, 200)
        vertices[2] = (vertices[2] - min(vertices[2])) * 200.0/(max_z + 1)

        # Round and cast vertices to integers
        vertices = np.round(vertices).astype(int)

        # Clip vertices values to [0, 200)
        vertices = np.clip(vertices, 0, 199)
        vertices = vertices.T

        # Create voxel space
        voxels = np.zeros((200, 200, 200), dtype='int')

        for row in vertices:
            x_coord = row[0]
            y_coord = row[1]
            z_coord = row[2]
            voxels[x_coord][y_coord][z_coord] = 1

        return voxels


''' Supply this function with the size of the batch you want.

    Returns a list of images, corresponding vertices, and corresponding voxels.
'''
def get_batch(size, include_transforms=False):
    vertices, voxels, images, transforms = [], [], [], []
    for _ in range(size):
        subset = ''
        index = np.random.randint(0, 3667)
        if index < AFW_size:
            subset = 'AFW'
        elif index < AFW_size + HELEN_size:
            subset = 'HELEN'
            index -= AFW_size
        elif index < AFW_size + HELEN_size + IBUG_size:
            subset = 'IBUG'
            index -= AFW_size + HELEN_size
        else:
            subset = 'LFPW'
            index -= AFW_size + HELEN_size + IBUG_size

        filedir = data_path + subset + '/'

        mats, jpgs, transforms = [], [], []
        for file in os.listdir(filedir):
            if file.endswith('.mat') and not file.endswith('_transform.mat'):
                mats.append(filedir + file)
                jpgs.append(filedir + file.replace('.mat', '.jpg'))
                transforms.append(filedir + file.replace('.mat', '') + '_transform.mat')

        data = scipy.io.loadmat(mats[index])
        transform = scipy.io.loadmat(transforms[index])
        vert = data['3D-vertices']
        vox = convert_to_voxels(vert)
        image = np.array(Image.open(jpgs[index]))

        vertices.append(vert)
        voxels.append(vox)
        images.append(image)
        transforms.append(transform)

    np_images = np.array(images)
    np_voxels = np.array(voxels)
    np_transforms = np.array(transforms)

    if include_transforms:
        return (np_images, np_voxels, transforms)
    return (np_images, np_voxels)


def get_visualization_batch(size):
    vertices, voxels, images, transforms = [], [], [], []
    for _ in range(size):
        subset = ''
        index = np.random.randint(0, 3667)
        if index < AFW_size:
            subset = 'AFW'
        elif index < AFW_size + HELEN_size:
            subset = 'HELEN'
            index -= AFW_size
        elif index < AFW_size + HELEN_size + IBUG_size:
            subset = 'IBUG'
            index -= AFW_size + HELEN_size
        else:
            subset = 'LFPW'
            index -= AFW_size + HELEN_size + IBUG_size

        filedir = data_path + subset + '/'
        originaldir = data_path + 'originals/' + subset + '/'

        mats, jpgs, transform_paths = [], [], []
        for file in os.listdir(filedir):
            if file.endswith('.mat') and not file.endswith('_transform.mat'):
                mats.append(filedir + file)
                jpgs.append(originaldir + file.replace('.mat', '_original.jpg'))
                transform_paths.append(filedir + file.replace('.mat', '') + '_transform.mat')

        data = scipy.io.loadmat(mats[index])
        transform = scipy.io.loadmat(transform_paths[index])
        vert = data['3D-vertices']
        vox = convert_to_voxels(vert)
        image = np.array(Image.open(jpgs[index]))

        vertices.append(vert)
        voxels.append(vox)
        images.append(image)
        transforms.append(transform)

    np_images = np.array(images)
    np_voxels = np.array(voxels)
    np_transforms = np.array(transforms)

    return (np_images, np_voxels, np_transforms)
