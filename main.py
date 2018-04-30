from Inference import predict
from colorized_voxels_demo import *
from preprocess import processFile
from PIL import Image
import numpy as np

from DataManager.manager import get_batch

def pipeline(filePath):
    # cropped, transform = processFile(filePath)

    cropped, _ = get_batch(1)
    cropped = cropped[0]

    #get voxels
    voxels = predict(cropped,loadFile=True)
    #visualize
    visualize_voxels_original(cropped, voxels)


if __name__ == '__main__':
    pipeline('face.jpg')
