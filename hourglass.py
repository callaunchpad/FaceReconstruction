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
    print(features["x"].shape)
    input_layer = tf.reshape(features["x"], [-1, 200, 200, 3])
    #construct the first half of our downsampling convolutional layers, going off of layer_details and pool_details
    conv_layers = [input_layer]
    # conv_input_shapes = [input_layer.get_shape()]
    last_layer = input_layer
    print("Generating hourglass conv layers")
    for (kernel_dim1, kernel_dim2, kernel_dim3, nfilters, padding, activation), (pool_dim1, pool_dim2, pool_dim3, stride) in zip(layer_details, pool_details):
        new_conv_layer = tf.layers.conv2d(
            inputs=last_layer,
            filters=nfilters,
            kernel_size=[kernel_dim1, kernel_dim2],
            padding=padding,
            activation=activation)
        # print("Original new conv layer shape:", new_conv_layer.shape)
        # new_conv_layer = tf.reshape(new_conv_layer, new_conv_layer.get_shape().as_list()[0:3]+[3,-1])
        # print(".")
        # print("Resized new conv layer shape:", new_conv_layer.shape)
        new_pool = tf.layers.max_pooling2d(inputs=new_conv_layer, pool_size=[pool_dim1, pool_dim2], strides=stride)
        # print("New pool shape:", new_pool.shape)

        #Each layer's size is dependent on their poolsize, and so we save these for upsampling later
        #conv_input_shapes.append(layers.shape)
        # layers.append(new_/conv_layer)
        conv_layers.append(new_pool)
        # conv_input_shapes.append(new_pool.get_shape())
        last_layer = new_pool
    # conv_input_shapes.pop()

    #upsample time!
    print("Generating hourglass upsample")
    print("shapes: %s" % [x.shape.as_list() for x in conv_layers])
    for i in range(len(conv_layers) - 1):
        #upsample by nearest neighbor
        # print(last_layer.shape)
        corresponding_layer = conv_layers[-(i+2)]
        print(corresponding_layer)

        upsampled_size = corresponding_layer.shape[1:3]
        print("Targe shape: %s" % upsampled_size.as_list())

        new_upsample_layer = tf.image.resize_nearest_neighbor(last_layer, size=upsampled_size)
        #if i == len(conv_input_shapes)-2:
        #new_upsample_layer = tf.reshape(new_upsample_layer, [-1, 200, 200, 1])
        #add residual layer
        residual_layer = residual_module(corresponding_layer)

        print("Upsampled layer shape: %s" % new_upsample_layer.shape)
        print("Residual input shape: %s" % conv_layers[-i].shape)
        print("Residual layer shape: %s" % residual_layer.shape)

        # if i == len(conv_input_shapes)-2:
        #     new_upsample_layer = tf.reshape(new_upsample_layer, layer_size)
        last_layer = tf.add(new_upsample_layer, residual_layer)
        # layers.append(new_upsample_layer)


    #finally, 200 1x1 convolutional layers and we're done
    new_conv_layer = tf.layers.conv2d(
        inputs=last_layer,
        filters=200,
        kernel_size=[1,1],
        activation=None)
    # new_conv_layer = tf.reshape(new_conv_layer, [-1, 200, 200, 200])
    print(new_conv_layer.shape)
    last_layer = new_conv_layer
    #return our last layer, the output
    return last_layer

#given an input size and output size, find the right kernel size for CNN
def get_kernel_size(layer_in, layer_out, padding="none"):
    if padding=="none":
        return (layer_in[0]-layer_out[0]+1, layer_in[1]-layer_out[1]+1)

def get_layer_size(layer_in, kernel, padding="none"):
    if padding == "none":
        return (layer_in[0]-kernel[0]+1,layer_in[1]-kernel[1]+1,layer_in[2]-kernel[2]+1)

def hourglass_model_fn(features, labels, mode):
    #layers = [(200-40*i, 200-40*i) for i in range(5)]
    #kernels = [get_kernel_size(layers[i], layers[i+1]) for i in range(len(layers)-1)]
    #filters = [1 for i in range(len(layers)-1)]
    layers = [(200, 200, 3), (125, 125, 3), (50, 50, 3), (4, 4, 3)]
    kernels = [(4, 4, 3), (4, 4, 3), (4, 4, 3)]
    filters = [3 for i in range(len(layers)-1)]
    padding = ["valid" for i in range(len(layers)-1)]
    activation = [tf.nn.relu for i in range(len(layers)-1)]

    layer_details = [(kernels[i][0], kernels[i][1], kernels[i][2], filters[i], padding[i], activation[i]) for i in range(len(layers)-1)]
    residual_model = lambda x : x
    pool_details = [(73, 73, 1, 1), (73, 73, 1, 1), (44, 44, 1, 1)]

    hourglass_model = get_hourglass(features, layer_details, pool_details, residual_model)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return hourglass_model

    labels = tf.reshape(labels, [-1, 200, 200, 200])
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hourglass_model, labels=labels), name= 'cross_entropy_loss')
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
        batch_size=2,
        num_epochs=None,
        shuffle=True)
    facelift.train(input_fn=train_input_fn, steps=20000)
    return facelift


train_data = np.array([np.random.rand(200, 200, 3) for i in range(2)]).astype('float32')
train_labels = np.array([np.random.rand(200, 200, 200) for i in range(2)]).astype('float32')
model_path = "hourglass_util/"
train_model(model_path, train_data, train_labels)
