import tensorflow as tf
import numpy as np
import sklearn

'''
Our input layer is 200x200, and the smallest we get to is 4x4
Input
    layer_details: A list of tuples representing specifics about each convolutional layer, in the format
                   (kernel_dim1, kernel_dim2, nfilters, padding, activation)
    pool_details: A list of tuples representing specifics about each pooling layer, in the format
                  (pool_dim1, pool_dim2, stride)
    residual_module: A function that takes a layer and outputs its residual to be added
Output
    An untrained hourglass network
'''
def get_hourglass(features, layer_details, pool_details, residual_module):
    #our input is 200x200, a regular 2D image
    input_layer = tf.reshape(features["x"], [-1, 200, 200, 1])

    #construct the first half of our downsampling convolutional layers, going off of layer_details and pool_details
    layers = [input_layer]
    upsample_sizes = []
    for (kernel_dim1, kernel_dim2, nfilters, padding, activation), (pool_dim1, pool_dim2, stride) in zip(layer_details, pool_details):
        new_conv_layer = tf.layers.conv2d(
            inputs=layers[-1],
            filters=nfilters,
            kernel_size=[kernel_dim1, kernel_dim2],
            padding=padding,
            activation=activation)
        new_pool = tf.layers.max_pooling2d(inputs=new_conv_layer, pool_size=[pool_dim1, pool_dim2], strides=stride)
        #Each layer's size is dependent on their poolsize, and so we save these for upsampling later
        upsample_sizes.append(tf.size(new_pool))
        layers.append(new_conv_layer)
        layers.append(new_pool)

    #upsample time!
    for i, layer_size in enumerate(reversed(upsample_sizes[:-1])):
        #upsample by nearest neighbor
        print(tf.shape(layers[-1]))
        new_upsample_layer = tf.image.resize_nearest_neighbor(layers[-1],size=layer_size)
        #add residual layer
        residual_layer = residual_module(layers[-1*(3*i+1)])
        new_upsample_layer = tf.add(new_upsample_layer, layers[-1*(3*i+1)])
        layers.append(new_upsample_layer)

    #finally, 2 1x1 convolutional layers and we're done
    for i in range(2):
        new_conv_layer = tf.layers.conv2d(
            inputs=layers[-1],
            filters=1,
            kernel_size=[1,1],
            activation=None)
        layers.append(new_conv_layer)
    #return our last layer, the output
    return layers[-1]

#given an input size and output size, find the right kernel size for CNN
def get_kernel_size(layer_in, layer_out, padding="none"):
    if padding=="none":
        return (layer_in[0]-layer_out[0]+1, layer_in[1]-layer_out[1]+1)

def hourglass_model_fn(features, labels, mode):
    layers = [(200, 200), (125, 125), (75, 75), (25, 25), (4, 4)]
    kernels = [get_kernel_size(layers[i], layers[i+1]) for i in range(len(layers)-1)]
    filters = [10 for i in range(len(layers)-1)]
    padding = ["same" for i in range(len(layers)-1)]
    activation = [tf.nn.relu for i in range(len(layers)-1)]

    layer_details = [(kernels[i][0], kernels[i][1], filters[i], padding[i], activation[i]) for i in range(len(layers)-1)]
    residual_model = lambda x : x
    pool_details = [(1, 1, 1) for i in range(len(layers)-1)]

    hourglass_model = get_hourglass(features, layer_details, pool_details, residual_model)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return hourglass_model

    loss = tf.losses.mean_squared_error(labels=labels,predictions=hourglass_model)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=hourglass_model)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

#given a path for saving progress for our model, training and label data, returns trained model.
#this is where most of the training will take effect
def train_model(path, train_data, train_labels):
    facelift = tf.estimator.Estimator(model_fn=hourglass_model_fn, model_dir=path)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    facelift.train(input_fn=train_input_fn, steps=20000)
    return facelift


train_data = np.array([np.random.rand(200, 200) for i in range(1000)]).astype('float32')
train_labels = np.array([np.random.rand(200, 200) for i in range(1000)]).astype('float32')
model_path = "hourglass_util/"

train_model(model_path, train_data, train_labels)






