'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.

'''
from manager import get_batch
from colorized_voxels_demo import subsample
from manager import max_z, convert_to_voxels
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from skimage import measure

def writeOBJ(params):
    verts, faces, normals, values = params
    faces += 1
    #https://stackoverflow.com/questions/48844778/create-a-obj-file-from-3d-array-in-python
    with open('test.obj', 'w') as outputOBJ:
        for item in verts:
            outputOBJ.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in normals:
            outputOBJ.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in faces:
            outputOBJ.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))  

images, vertex_lst, voxel_lst = get_batch(1)
vertices = vertex_lst[0]
print(vertices)
image = images[0]
voxels = voxel_lst[0]


verts, faces, normals, values = measure.marching_cubes_lewiner(voxels, 0)
writeOBJ((verts, faces, normals, values))

#dont plot it its laggy

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
#                 linewidth=0.2, antialiased=True)
# plt.show()

