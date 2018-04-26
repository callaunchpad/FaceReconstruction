from Inference import predict
from colorized_voxels_demo import visualize_voxels
from preprocess import processFile
from PIL import Image
import numpy as np


def pipeline(filePath):
    cropped, transform = processFile(filePath)
    #get voxels
    voxels = predict(cropped,loadFile=True)
    #visualize
    visualize_voxels(cropped, voxels)


if __name__ == '__main__':
    pipeline('face.jpg')
