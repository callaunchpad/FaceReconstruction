import numpy as np
#
# Input x,y,z for point we are manipulating. Choose rotation axis from bottom 3 Functions.
# Put translation in X, Y, Z directions.
#
def transform(x,y,z,rotation_matrix=rotate_x_axis(np.pi/2), translation_x=0, translation_y=0, translation_z=0):
    translation = np.array([[translation_x], [translation_y], [translation_z]])
    matrix = np.hstack((rotation_matrix,translation))
    matrix = np.vstack((matrix, [0,0,0,1]))
    result = matrix.dot(np.array([x,y,z,1]))
    return result[:3]


def rotate_x_axis(theta):
    matrix = np.array([[1,0,0],
                       [0,np.cos(theta),-np.sin(theta)],
                       [0,np.sin(theta),np.cos(theta)]])

    return matrix

def rotate_y_axis(theta):
    matrix = np.array([[np.cos(theta),0,np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta),0,np.cos(theta)]])
    return matrix

def rotate_z_axis(theta):
    matrix = np.array([[np.cos(theta),-np.sin(theta),0],
                       [np.sin(theta),np.cos(theta,0)],
                       [0,0,1]])
    return matrix

print(transform(1,0,0,rotate_y_axis(0),1,0,0))