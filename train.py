from hourglass import *
import tensorflow as tf
import numpy as np
from DataManager.manager import get_batch

def get_model(input, name='hourglass'):

    layer_dims = list(reversed(range(20, 200, 10)))

    hourglass = get_hourglass(input, layer_dims, output_size=256)
    hourglass = get_hourglass(hourglass, layer_dims, output_size=200)
    return tf.identity(hourglass, name=name)


#given a path for saving progress for our model, training and label data, returns trained model.
#this is where most of the trgit aining will take effect
def train_model(batch_size, iterations, load=False):
    input = tf.placeholder(tf.float32, name="input", shape=(None, 200, 200, 3))
    labels = tf.placeholder(tf.float32, name="labels", shape=(None, 200, 200, 200))
    hourglass_model = get_model(input, name='hourglass')

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=hourglass_model, labels=labels)
    loss = tf.reduce_mean(cross_entropy, name= 'cross_entropy_loss')
    # classified = (tf.sign(hourglass_model) + 1) / 2
    # accuracy = tf.reduce_mean(tf.abs(classified - labels))

    saver = tf.train.Saver()

    adam_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
    images, voxels = get_batch(batch_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if load:
            saver = tf.train.import_meta_graph('./models/chkpt.meta')
            saver.restore(sess, './models/chkpt')
            graph = tf.get_default_graph()
            input = graph.get_tensor_by_name("input:0")
            labels = graph.get_tensor_by_name("labels:0")
            model = graph.get_tensor_by_name("hourglass:0")
            loss = graph.get_tensor_by_name("cross_entropy_loss:0")

        for i in range(iterations):
            print("Iteration %i" % i)
            # images, voxels = get_batch(batch_size)
            feed_dict = {input: images, labels: voxels}
            try:
                train_step = adam_step
                sess.run(train_step, feed_dict=feed_dict)
            except ValueError:
                print("Random error optimizing, don't know what's wrong. Just skipping this epoch.")
                continue
            if i % 1 == 0:
                try:
                    err = sess.run(loss, feed_dict=feed_dict)
                    print("Loss: %i, %f " % (i, err))
                    # print("Accuracy: %f" % accuracy)
                except ValueError:
                    print("Random error calculating loss, don't know what's wrong. Just skipping this epoch.")
                    pass
            #save our sess every 100 iterations
            if (i % 10 == 0):
                saver.save(sess, './models/chkpt')


    return hourglass_model

if __name__ == "__main__":
    # model_path = "hourglass_util/"
    train_model(batch_size=2, iterations=2000)
