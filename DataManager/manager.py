# Program to convert to the numpy matrix into voxels - 3D window with
import numpy as np
import scipy
import scipy.io as scio
import os

max_z = 248

AFW_size = 313
HELEN_size = 2253
IBUG_size = 105
LFPW_size = 996

# Image paths
data_path = './300W-3D/'
subsets = ['AFW', 'HELEN', 'IBUG', 'LFPW']

data_path = '../preprocessed/'

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

    Returns a list of vertics and corresponding voxels.
'''
def get_batch(size):
    vertices, voxels = [], []
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

        matdir = data_path + subset + '/'

        mats = []
        for file in os.listdir(matdir):
            if file.endswith('.mat'):
                mats.append(matdir + file)

        data = scipy.io.loadmat(mats[index])
        vert = data['3D-vertices']
        vox = convert_to_voxels(vert)

        vertices.append(vert)
        voxels.append(vox)

    return (vertices, voxels)
