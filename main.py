from Inference import predict
from colorized_voxels_demo import visualize_voxels
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
    visualize_voxels(cropped, voxels)


if __name__ == '__main__':
    pipeline('face.jpg')
