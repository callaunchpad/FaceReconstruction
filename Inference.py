import tensorflow as tf
import numpy as np
from train import get_model
from PIL import Image
from DataManager.manager import convert_to_voxels
import scipy

sess = tf.Session()
input = tf.placeholder(tf.float32, name="input", shape=(None, 200, 200, 3))
labels = tf.placeholder(tf.float32, name="labels", shape=(None, 200, 200, 200))
model = get_model(input, name="hourglass")
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, './models/chkpt')

def predict(filepath, loadFile=False):
    #get the input
    input = tf.placeholder(tf.float32, name="input", shape=(None, 200, 200, 3))
    if loadFile:
        image = np.array(filepath)
    else:
        image = np.array(Image.open(filepath))
    with tf.Session() as sess:
        # hourglass_model = get_model(input)
        # #load our variables
        # saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph('./models/chkpt.meta')
        # saver.restore(sess,tf.train.latest_checkpoint('./models/'))
        voxels = sess.run(model, feed_dict = {input: [image]})[0]
        return np.where(voxels > 0, 1, 0)



if __name__ == '__main__':
    predict('./300W_LP/AFW/AFW_134212_1_1')
