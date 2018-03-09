import numpy as np
import sys
sys.path.insert(0, 'DataManager')
fm = __import__('3D-Face-FileManager')

# Input x,y,z for point we are manipulating. Choose rotation axis from bottom 3 Functions.
# Put translation in X, Y, Z directions.
#


def rotate_x_axis(theta):
    matrix = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta), np.cos(theta)]])

    return matrix


def rotate_y_axis(theta):
    matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
    return matrix


def rotate_z_axis(theta):
    matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta, 0)],
                       [0, 0, 1]])
    return matrix


def transform(x, y, z, rotation_matrix=rotate_x_axis(np.pi / 2), translation_x=0, translation_y=0, translation_z=0):
    translation = np.array([[translation_x], [translation_y], [translation_z]])
    matrix = np.hstack((rotation_matrix, translation))
    matrix = np.vstack((matrix, [0, 0, 0, 1]))
    result = matrix.dot(np.array([x, y, z, 1]))
    return result[:3]

# Transforms vertices in the form [[x...][y...][z...]]


def transform_vertices(vertices, rotation_matrix=rotate_x_axis(np.pi / 2), translation_x=0, translation_y=0, translation_z=0):
    return list(zip(*[transform(x, y, z, rotation_matrix, translation_x, translation_y, translation_z) for x, y, z in zip(*vertices)]))

# Centers a list of vertices around the origin


def center_vertices(vertices):
    center_x = (max(vertices[0]) + min(vertices[0])) / 2
    center_y = (max(vertices[1]) + min(vertices[1])) / 2
    center_z = (max(vertices[2]) + min(vertices[2])) / 2
    return transform_vertices(vertices, rotate_y_axis(0), -center_x, -center_y, -center_z)


# print(transform(1, 0, 0, rotate_y_axis(0), 1, 0, 0))
# print(transform_vertices(((1, 1, 1, 1), (1, 2, 3, 4), (1, 2, 3, 4)), rotate_y_axis(0), 1, 0, 0))
# print(center_vertices(((1, 1, 1, 1), (1, 2, 3, 4), (1, 2, 3, 4))))
# fm.plot_verticies(center_verticesfm.get_datum(next(fm.enumerate_data_paths()))[1]))
