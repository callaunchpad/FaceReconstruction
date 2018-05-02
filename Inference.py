import tensorflow as tf
import numpy as np
from train import get_model
from PIL import Image
from DataManager.manager import convert_to_voxels
import scipy
from DataManager.surface_face_march import voxelToOBJ
from DataManager.manager import get_batch

sess = tf.Session()
# inpt = tf.placeholder(tf.float32, name="input", shape=(None, 200, 200, 3))
# labels = tf.placeholder(tf.float32, name="labels", shape=(None, 200, 200, 200))
# model = get_model(inpt, name="hourglass")
# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())
# saver = tf.train.Saver()
# print('reee')
# tf.reset_default_graph()

# path = tf.train.get_checkpoint_state('./models/chkpt')
saver = tf.train.import_meta_graph('./models/chkpt.meta')
# saver.restore(sess, path.mode.checkpointpath)
saver.restore(sess, './models/chkpt')
graph = tf.get_default_graph()
# print('resotred')
inpt = graph.get_tensor_by_name("input:0")
labels = graph.get_tensor_by_name("labels:0")
model = graph.get_tensor_by_name("hourglass:0")
loss = graph.get_tensor_by_name("cross_entropy_loss:0")


def predict(filepath, loadFile=False):
    #get the input
    # if loadFile:
    #     image = np.array(filepath)
    # else:
    #     image = np.array(Image.open(filepath))

    crop, vox3 = get_batch(1)
    image = crop[0]
    label_vox = vox3[0]

    # hourglass_model = get_model(input)
    # #load our variables
    # saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph('./models/chkpt.meta')
    # saver.restore(sess,tf.train.latest_checkpoint('./models/'))
    # print(type(image))
    los2 = costTF(model, labels)
    voxels, los, loss2 = sess.run([model, loss, los2], feed_dict = {inpt: [image], labels:[label_vox]})
    voxels = voxels[0]
    # los = sess.run(loss, feed_dict = {inpt: [image], labels:[label_vox]})
    # los2 = sess.run(costTF(voxels, labels), feed_dict = {inpt: [image], labels:[label_vox]})
    print('loss is', los)
    print('loss2 is', loss2)
    print(np.min(voxels), np.mean(voxels), np.max(voxels))
    print(np.sum(label_vox))
    # print(np.sum(np.square(sigmoid(voxels) - label_vox)))
    print("loss2 numpy", cost(voxels, label_vox))
    print(voxels)
    # return np.round(sigmoid(voxels))
    return np.where(voxels > 0, 1, 0)
    # return np.round(voxels)


def costTF(x, z):
    temp = z * -1 * tf.log(tf.sigmoid(x)) + (z - 1) * tf.log(1 - tf.sigmoid(x))
    loss = tf.reduce_mean(temp)
    return loss

def cost(x, z):
    # log = np.log
    # return z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    temp = z * -np.log(sigmoid(x)) + (1 - z) * -np.log(1 - sigmoid(x))
    # temp = np.max(x, 0) - x * z + np.log(1 + np.exp(-np.abs(x)))
    return np.sum(temp)

sigmoid = lambda x: 1/(1+np.exp(-x))

if __name__ == '__main__':
    voxee = predict('')#./preprocessed/AFW/261068_2.jpg
    voxelToOBJ(voxee, "output7")
