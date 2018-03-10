# Program to convert to the numpy matrix into voxels - 3D window with
import numpy as np
import scipy
import scipy.io as scio



def convert_to_voxels(file):
    mat = scio.loadmat(file)['vertices']
    mat = np.transpose(mat)

    voxels = np.zeros((300, 300, 300), dtype = 'float32')

    #voxels = np.transpose(voxels)
    voxels = np.rint(voxels)

    for row in mat:
        x_coord = row[0]
        y_coord = row[1]
        z_coord = row[2]

        voxels[x_coord][y_coord][z_coord] = 1

    return voxels

''' 
testmat = np.zeros((3,3))
testmat.append([1.5, 2.3, 35.6])

output = np.round(testmat)
print(output)
'''
