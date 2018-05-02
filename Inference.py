import tensorflow as tf
import numpy as np
import model
from PIL import Image
from DataManager.manager import convert_to_voxels
import scipy
from DataManager.surface_face_march import voxelToOBJ
from DataManager.manager import get_batch
from colorized_voxels_demo import visualize_voxels_cropped

sess = tf.Session()

input, label, model, loss, _ = model.load_model(sess=sess)

def predict(filepath, loadFile=False):
    #get the input
    # if loadFile:
    #     image = np.array(filepath)
    # else:
    #     image = np.array(Image.open(filepath))

    crop, vox_label = get_batch(1)
    image = crop[0]
    vox_label = vox_label[0]

    voxels, err = sess.run([model, loss], feed_dict = {input: [image], label:[vox_label]})
    voxels = voxels[0]

    # voxels = np.where(voxels > 0, 1, 0)
    visualize_voxels_cropped(None, vox_label, save=True, file_prefix="images/True")
    for i in range(7):
        visualize_voxels_cropped(None, np.where(voxels > -i, 1, 0), save=True, file_prefix="images/" + str(i))

    return voxels, err, vox_label


sigmoid = lambda x: 1/(1+np.exp(-x))

voxee, err, vox_label = predict('')


# if __name__ == '__main__':
#     voxee = predict('./preprocessed/AFW/261068_2.jpg')#./preprocessed/AFW/261068_2.jpg
#     visualize_voxels_cropped(None, voxee, False)
    # voxelToOBJ(voxee, "output7")
