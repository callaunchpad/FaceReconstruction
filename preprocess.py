import sys
import os
import dlib
import glob
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from skimage import io
from PIL import Image
from scipy.io import loadmat
from scipy.io import savemat
from load_labels import process_3D_labels, load_3D_labels


"""Preprocessing script for faces and 3D mesh data.

Script iterates through images in the 300W-3D dataset and
performs the following:

1.  Detects the face in each image and crops it to contain 
    only the face.
2.  Resizes and upsamples the image to (200, 200).
3.  Transforms and crops the associated 3D mesh to align 
    with the cropped image.

You should have the 300W-3D and 300W-3D-Face datasets in
your folder before running.

Usage:
    $ python preprocessing.py

"""


#####################################
### File Paths and Face Detectors ###
#####################################

predictor_path = "DetectorFiles/shape_predictor_68_face_landmarks.dat"
cnn_path = "DetectorFiles/mmod_human_face_detector.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_path)

# Image paths
data_path = './300W-3D/'
subsets = ['AFW', 'HELEN', 'IBUG', 'LFPW']

# Set to True if you want to show show cropped images 
# and plot transformed meshes afterwards
show_examples = True



########################
### Helper Functions ###
########################

def subsample(vertices, rate):
        """ Returns a subset of the vertices given. Used for plotting transformed meshes
        over cropped images in the section 'Plotting Examples'.

        Adjust the subsampling rate to plot sparser/thicker meshes on your image.
        """
        return [vertices[i] for i in range(len(vertices)) if i%rate == 0]

def computeMatrixAndFace(face_path):
    dimensions = getBounding(face_path)
    if dimensions:
        cropped_img = cropFace(face_path, dimensions)
        A, b = getTransform(dimensions)
        print(A)
        print(b)

def getTransform(dimensions):
    x_max, y_max, x_min, y_min = dimensions
    transform = {}
    shift = np.array([[-x_min, -y_min, 1]]).T
    scale = np.diag([200/float(x_max - x_min), 200/float(y_max - y_min), 1])
    transform['A'] = scale
    transform['b'] = scale.dot(shift)
    return transform

def getBounding(face_path, use_cnn = False):
#     print("Processing file: {}".format(face_path))
    img = io.imread(face_path)

    dets = cnn_face_detector(img, 0) if use_cnn else detector(img, 0)

    for k, d in enumerate(dets):

        shape = predictor(img, d.rect) if use_cnn else predictor(img, d)
        
        x_min = y_min = float('inf')
        x_max = y_max = -float('inf')
        for i in shape.parts():
            x_max = max(i.x, x_max)
            y_max = max(i.y, y_max)
            x_min = min(i.x, x_min)
            y_min = min(i.y, y_min)
        return (x_max, y_max, x_min, y_min)
    
def saveFace(output_path, pil_img):
    pil_img.save(output_path)

def cropFace(face_path, dimensions):
    x_max, y_max, x_min, y_min = dimensions
    crop_rectangle = (x_min, y_min, x_max, y_max)
    im = Image.open(face_path)
    cropped_im_res = im.crop(crop_rectangle).resize((200, 200), Image.ANTIALIAS)
    return cropped_im_res



############################
### Preprocessing Script ###
############################

# Create cropped image folders
crop_folder = './preprocessed/'
for subset in subsets:
    if not os.path.exists(crop_folder + subset):
        os.makedirs(crop_folder + subset)

# Create processed mesh folders
mesh_out_folder = './preprocessed/'
for subset in subsets:
    if not os.path.exists(mesh_out_folder + subset):
        os.makedirs(mesh_out_folder + subset)

for subset in subsets:
    mypath = data_path + subset + '/'
    filepaths = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f[-4:] == '.jpg']
    
    for i in range(len(filepaths)):
        f = filepaths[i]
        face_path = data_path + subset + '/' + f
        landmark_path = face_path[:-4] + '.mat'
        mesh_path = './300W-3D-Face/' + subset + '/' + f[:-4] + '.mat'
        mesh_out_path = mesh_out_folder + subset + '/' + f[:-4] + '.mat'
        cropped_path = crop_folder + subset + '/' + f
        
        dimensions = getBounding(face_path)
        if dimensions:
            cropped_img = cropFace(face_path, dimensions)
            transform = getTransform(dimensions)
            
            process_3D_labels(landmark_path, mesh_path, transform['A'], transform['b'], 200, 200, mesh_out_path)
            
            saveFace(cropped_path, cropped_img)

        if not i % 50:
            print("Processing " + subset + " image {} of {}".format(i, len(filepaths)))



#########################
### Plotting Examples ###
#########################

""" Displays some examples of cropped images and plots their transformed meshes."""

if show_examples:

    samples = ['AFW/4538917191_5', 'HELEN/3004338997_1', 'IBUG/image_027', 'LFPW/image_train_0166']

    for sample in samples:
        face_path = crop_folder + sample + '.jpg'
        mesh_out_path = mesh_out_folder + sample + '.mat'

        img = Image.open(face_path)
        mesh_dict = loadmat(mesh_out_path)
        vertices = mesh_dict['3D-vertices']

        plt.figure()
        plt.imshow(img)
        plt.plot(subsample(vertices[0], 30), subsample(vertices[1], 30), 'g.')            
        
    plt.show()

