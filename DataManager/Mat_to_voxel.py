# Program to convert to the numpy matrix into voxels - 3D window with
import numpy as np
import scipy
import scipy.io as scio

max_z = 248

def convert_to_voxels(vertices):

    # Scale z-coordinates to fit in [0, 200)
    vertices[2] = (vertices[2] - min(vertices[2])) * 200.0/249.0

    # Round and cast vertices to integers
    # vertices = np.round(scio.loadmat(file)['3D-vertices']).astype(int)
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
