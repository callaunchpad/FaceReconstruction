import scipy.io as scio
import numpy as np
import sys
sys.path.insert(0, 'DataManager')
fm = __import__('3D-Face-FileManager')


def load_3D_labels(landmark_path, mesh_path):
    vertices2D = scio.loadmat(landmark_path)['pt2d']
    vertices2D = np.matmul(np.eye(3)[:, 0:2], vertices2D)
    vertices3D = scio.loadmat(mesh_path)['Fitted_Face']
    transformation_matrix = np.eye(3)
    transformation_matrix[1][1] = -1
    vertices3D = transformation_matrix.dot(vertices3D)
    average2D_y = np.mean(vertices2D[1])
    average3D_y = np.mean(vertices3D[1])
    vertices3D[1] = vertices3D[1] - (average3D_y - average2D_y) * \
        np.ones(vertices3D[1].shape)
    return vertices3D

# Takes in vertices hortizontally stacked, a linear_transformation_matrix, and translation coordinations as a column vector (3x1)
# Applies the translation and then the transformation


def affine_transform(vertices, linear_transform, translation):
    vertices = vertices + translation
    vertices = linear_transform.dot(vertices)
    return vertices


def crop_vertices(vertices, width, height):
    # result = []
    # for item in vertices.T:
    #     if not (item[0] > width or item[0] < 0 or item[1] > height or item[1] < 0):
    #         result.append(item)
    # result = np.array(result).T
    # return result
    vertices = vertices.T[np.logical_and(
        vertices[0, :] >= 0, vertices[0, :] <= width)].T
    vertices = vertices.T[np.logical_and(
        vertices[1, :] >= 0, vertices[1, :] <= height)].T
    return vertices

# Takes path to the landmark file, mesh file, a linear_transformation_matrix,
# and translation coordinations as a column vector (3x1), bounding width, bounding height, output filename


def process_3D_labels(landmark_path, mesh_path, linear_transform, translation, width, height, file_name):
    vertices = load_3D_labels(landmark_path, mesh_path)
    vertices = affine_transform(vertices, linear_transform, translation)
    vertices = crop_vertices(vertices, width, height)
    vertices_dic = {'3D-vertices': vertices}
    scio.savemat(file_name, vertices_dic)


# vertices = load_3D_labels(
#     'D:\\Users\\willf\\Desktop\\Launchpad\\FaceReconstruction\\300W-3D\\HELEN\\232194_1.mat', 'D:\\Users\\willf\\Desktop\\Launchpad\\FaceReconstruction\\300W-3D-Face\\HELEN\\232194_1.mat')
# vertices = crop_vertices(vertices, 350, 350)
# fm.plot_vertices(vertices[:, ::20])
# process_3D_labels('D:\\Users\\willf\\Desktop\\Launchpad\\FaceReconstruction\\300W-3D\\HELEN\\232194_1.mat',
#                   'D:\\Users\\willf\\Desktop\\Launchpad\\FaceReconstruction\\300W-3D-Face\\HELEN\\232194_1.mat', np.eye(3), 0, 300, 300, 'output')
