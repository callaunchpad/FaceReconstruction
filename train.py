from hourglass import *
import tensorflow as tf
import numpy as np
from DataManager.manager import get_batch
import model



def train_model(batch_size, iterations, load=True):

    # Create model
    step_size = tf.placeholder(tf.float32, name="stepsize")
    input = tf.placeholder(tf.float32, name="input", shape=(None, 200, 200, 3))
    labels = tf.placeholder(tf.float32, name="labels", shape=(None, 200, 200, 200))
    hourglass_model = model.get_model(input, name='hourglass')
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=hourglass_model, labels=labels)
    loss = tf.reduce_mean(cross_entropy, name= "cross_entropy_loss")
    adam_step = tf.train.AdamOptimizer(step_size, name="optimizer").minimize(loss)
    saver = tf.train.Saver()

    #Overfit to only this batch for now
    # images, voxels = get_batch(batch_size)

    with tf.Session() as sess:
        if load:
            input, labels, hourglass_model, loss, step_size, adam_step = model.load_model(sess=sess)
            print("Successfully loaded saved file")
        else:
            sess.run(tf.global_variables_initializer())
            print("Intialized model with random variables")

        for i in range(iterations):
            images, voxels = get_batch(batch_size)
            feed_dict = {input: images, labels: voxels, step_size: 1e-4}

            try:
                train_step = adam_step
                err, _ = sess.run([loss, train_step], feed_dict=feed_dict)
                print("Loss: %i, %f " % (i, err))
            except ValueError:
                print("Random error optimizing, don't know what's wrong. Just skipping this epoch.")
                continue

            #save our sess every 100 iterations
            if (i % 10 == 0):
                saver.save(sess, './models/chkpt')

    return hourglass_model

if __name__ == "__main__":
    # model_path = "hourglass_util/"
    train_model(batch_size=10, iterations=5000, load=False)
