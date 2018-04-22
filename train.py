from hourglass import *
import tensorflow as tf
import numpy as np

def get_model(input):
    layers = [(200, 200, 3), (125, 125, 3), (50, 50, 3), (4, 4, 3)]
    kernels = [(4, 4, 3), (4, 4, 3), (4, 4, 3)]
    filters = [3 for i in range(len(layers)-1)]
    padding = ["valid" for i in range(len(layers)-1)]
    activation = [tf.nn.relu for i in range(len(layers)-1)]
    
    layer_details = [(kernels[i][0], kernels[i][1], kernels[i][2], filters[i], padding[i], activation[i]) for i in range(len(layers)-1)]
    #residual_model = lambda x : x
    residual_model = resBlock
    pool_details = [(73, 73, 1, 1), (73, 73, 1, 1), (44, 44, 1, 1)]
    

    return get_hourglass(input, layer_details, pool_details, residual_model)


#given a path for saving progress for our model, training and label data, returns trained model.
#this is where most of the training will take effect
def train_model(batch_size, iterations):
    input = tf.placeholder(tf.float32, name="input", shape=(None, 200, 200, 3))
    labels = tf.placeholder(tf.float32, name="labels", shape=(None, 200, 200, 200))
    hourglass_model = get_model(input)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hourglass_model, labels=labels), name= 'cross_entropy_loss')
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            print("Iteration %i" % i)
            images, voxels = get_batch(batch_size)
            err, _ = sess.run([loss, train_step], feed_dict = {input: images, labels: voxels})
            #save our sess every 100 iterations
            if (i % 100 == 0):
                print("Error:" , err)
                saver.save(sess, 'hourglass_model_sess')


    return hourglass_model

if __name__ == "name":
    train_data = np.array([np.random.rand(200, 200, 3) for i in range(2)]).astype('float32')
    train_labels = np.array([np.random.rand(200, 200, 200) for i in range(2)]).astype('float32')
    model_path = "hourglass_util/"


    train_model(batch_size=100, iterations=1000)
