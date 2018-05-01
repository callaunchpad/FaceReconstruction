from Inference import predict
from colorized_voxels_demo import *
from preprocess import processFile
from PIL import Image
import numpy as np

from DataManager.manager import get_batch

def pipeline(filePath):
    # cropped, transform = processFile(filePath)

    cropped, label = get_batch(10)
    cropped = cropped
    label = label

    print(cropped.shape)

    #get voxels
    voxels = predict(cropped,loadFile=True)

    import tensorflow as tf
    predicted = tf.placeholder(tf.float32, name="input", shape=(None, 200, 200, 200))
    labels = tf.placeholder(tf.float32, name="labels", shape=(None, 200, 200, 200))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predicted), name= 'cross_entropy_loss')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(label.shape)
    print(voxels.shape)
    print(sess.run(loss, feed_dict={labels: label, predicted: voxels}))


    #visualize
    visualize_voxels_cropped(cropped, voxels)


if __name__ == '__main__':
    pipeline('face.jpg')
