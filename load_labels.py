import sys
import numpy as np
import scipy.io as scio
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

# Takes in vertices as a column vector, a linear_transformation_matrix [A], and translation coordinations as a column vector [b] (3x1)
# Applies transformation Ax + b

def affine_transform(vertices, linear_transform, translation):
    vertices = linear_transform.dot(vertices)
    vertices += translation
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

# Takes path to the landmark file, mesh file, a linear_transformation_matrix [A],
# and translation coordinations as a column vector [b] (3x1), bounding width, bounding height, output path

def process_3D_labels(landmark_path, mesh_path, linear_transform, translation, width, height, out_path):
    vertices = load_3D_labels(landmark_path, mesh_path)
    vertices = affine_transform(vertices, linear_transform, translation)
    vertices = crop_vertices(vertices, width, height)
    if vertices.size:
        vertices_dic = {'3D-vertices': vertices}
        scio.savemat(out_path, vertices_dic)
