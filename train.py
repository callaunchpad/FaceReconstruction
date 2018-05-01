from hourglass import *
import tensorflow as tf
from DataManager.manager import get_batch

def get_model(input, name='hourglass'):
    layers = [(200, 200, 3), (125, 125, 3), (50, 50, 3), (4, 4, 3)]
    kernels = [(4, 4, 3), (4, 4, 3), (4, 4, 3)]
    filters = [256 for i in range(len(layers)-1)]
    padding = ["valid" for i in range(len(layers)-1)]
    activation = [tf.nn.elu for i in range(len(layers)-1)]

    layer_details = [(kernels[i][0], kernels[i][1], kernels[i][2], filters[i], padding[i], activation[i]) for i in range(len(layers)-1)]
    #residual_model = lambda x : x
    residual_model = resBlock
    pool_details = [(73, 73, 1, 1), (73, 73, 1, 1), (44, 44, 1, 1)]

    hourglass = get_hourglass(input, layer_details, pool_details, residual_model)
    return tf.identity(hourglass, name=name)


#given a path for saving progress for our model, training and label data, returns trained model.
#this is where most of the trgit aining will take effect
def train_model(batch_size, iterations):
    input = tf.placeholder(tf.float32, name="input", shape=(None, 200, 200, 3))
    labels = tf.placeholder(tf.float32, name="labels", shape=(None, 200, 200, 200))
    hourglass_model = get_model(input, name='hourglass')

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=hourglass_model, labels=labels)
    loss = tf.reduce_mean(cross_entropy, name= 'cross_entropy_loss')

    saver = tf.train.Saver()

    first_optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
    images, voxels = get_batch(batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_step = first_optimizer
        for i in range(iterations):
            print("Iteration %i" % i)
            # images, voxels = get_batch(batch_size)
            feed_dict = {input: images, labels: voxels}
            try:
                sess.run(train_step, feed_dict=feed_dict)
            except ValueError:
                print("Random error optimizing, don't know what's wrong. Just skipping this epoch.")
                continue
            if i % 5 == 0:
                try:
                    err = sess.run(loss, feed_dict=feed_dict)
                    print("Loss: %i, %f " % (i, err))
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
