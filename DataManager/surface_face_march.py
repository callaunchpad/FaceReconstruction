'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.

'''
# from colorized_voxels_demo import subsample
from mpl_toolkits.mplot3d import Axes3D
# from manager import max_z
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from skimage import measure
from scipy.spatial import Delaunay, ConvexHull

def writeOBJ(params, name):
    verts, faces, normals, values = params
    faces += 1
    #https://stackoverflow.com/questions/48844778/create-a-obj-file-from-3d-array-in-python
    with open(name + '.obj', 'w') as outputOBJ:
        for item in verts:
            outputOBJ.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in normals:
            outputOBJ.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in faces:
            outputOBJ.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))  

def voxelToOBJ(voxel_inpt, name):
    space = 1
    volume = np.ascontiguousarray(voxel_inpt, np.float32)
    print(volume.min(), volume.max())
    verts, faces, normals, values = measure.marching_cubes_lewiner(voxel_inpt, level=0, spacing=(space, space, space), step_size = 1)
    
    # #dont plot it its laggy
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
    #                 linewidth=0.2, antialiased=True)
    # plt.show()

    writeOBJ((verts, faces, normals, values), name)

def write_stl_ascii(filename, points, faces):
    pts = points[faces]
    normals = compute_normals(pts)
    print(pts)

    with open(filename, 'wb') as fh:
        fh.write('solid\n'.encode('utf-8'))

        for local_pts, normal in zip(pts, normals):
            # facet normal 0.455194 -0.187301 -0.870469
            #  outer loop
            #   vertex 266.36 234.594 14.6145
            #   vertex 268.582 234.968 15.6956
            #   vertex 267.689 232.646 15.7283
            #  endloop
            # endfacet
            fh.write(' facet normal {} {} {}\n'.format(*normal).encode('utf-8'))
            fh.write('  outer loop\n'.encode('utf-8'))
            for pt in local_pts:
                fh.write('   vertex {} {} {}\n'.format(*pt).encode('utf-8'))
            fh.write('  endloop\n'.encode('utf-8'))
            fh.write(' endfacet\n'.encode('utf-8'))

        fh.write('endsolid\n'.encode('utf-8'))

def voxelToSTL(voxel_inpt, name):
    verts, faces, normals, values = measure.marching_cubes_lewiner(voxel_inpt, 0)
    # import meshio
    # cells = {'triangle': faces}
    write_stl_ascii(name + ".stl", verts, faces)

def compute_normals(pts):
    normals = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    nrm = np.sqrt(np.einsum('ij,ij->i', normals, normals))
    normals = (normals.T / nrm).T
    return normals

def convert_to_fatvoxels(vertices, size):
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
            
            for r in range(size):
                for c in range(size):
                    idX = x_coord + int(r - size/2)
                    idY = y_coord + int(c - size/2)
                    if (idX >= 0 and idX < 200) and (idY >= 0 and idY < 200):
                        voxels[idX][idY][z_coord] = 1
            

        return voxels

if __name__ == '__main__':
    from manager import get_batch
    images, vertex_lst, voxel_lst = get_batch(1)
    vertices = vertex_lst[0]
    print("vertx shape", vertices.shape)
    print(vertices.T)
    image = images[0]
    voxels = voxel_lst[0]
    print(voxels.shape)
    fat_voxels = convert_to_fatvoxels(vertices, 4)

    # pts = vertices[0:2].T
    # print(pts.shape)
    # tri = Delaunay(pts)
    # pts = vertices.T
    # print(tri.simplices.shape)
    # normals = compute_normals(pts[tri.simplices])

    # hull = ConvexHull(vertices.T)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot(vertices[0], vertices[1], vertices[2], "ko")
    # print(hull.simplices.shape)
    # normals = compute_normals(vertices.T[hull.simplices])

    # for s in hull.simplices:
    #     s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        # ax.plot(vertices.T[s, 0], vertices.T[s, 1], vertices.T[s, 2], "r-")
    # plt.show()  

    # writeOBJ((vertices.T, hull.simplices, normals, None), 'test_5')
    voxelToOBJ(fat_voxels, 'test6')
    voxelToSTL(fat_voxels, 'test6')


   

