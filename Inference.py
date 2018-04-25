import tensorflow as tf
import numpy as np
from train import get_model
from PIL import Image
from DataManager.manager import convert_to_voxels
import scipy

def predict(filepath):
    #get the input
    input = tf.placeholder(tf.float32, name="input", shape=(None, 200, 200, 3))
    image = np.array(Image.open(filepath+'.jpg'))
    image = np.random.rand(200, 200, 3)
    print(image)
    with tf.Session() as sess:
        hourglass_model = get_model(input)
        #load our variables
        saver = tf.train.Saver()
        saver = tf.train.import_meta_graph('./models/chkpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./models/'))
        print(sess.run(hourglass_model, feed_dict = {input: [image]}))


predict('./300W_LP/AFW/AFW_134212_1_1')
