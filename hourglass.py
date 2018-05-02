import tensorflow as tf
import tensorflow.contrib.slim as slim

'''
RESBLOCK BOIZ
'''
def resBlock(x,channels=256,kernel_size=[3,3], activation=tf.nn.leaky_relu):
    result = tf.layers.conv2d(inputs=x, filters=channels/2, kernel_size=[1,1], activation=activation)
    result = tf.layers.conv2d(inputs=result, filters=channels/2, kernel_size=kernel_size, activation=activation)
    result = tf.layers.conv2d(inputs=result, filters=channels, kernel_size=[1,1], activation=activation)
    result = slim.conv2d(x,channels/2,kernel_size,activation_fn=activation)
    return x + result

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
def get_hourglass(input_layer, layer_dims, output_size=200):
    kernel_size = 4
    nfilters = 256
    activation = tf.nn.leaky_relu
    padding = "valid"
    stride = 1
    residual_module = resBlock

    #construct the first half of our downsampling convolutional layers, going off of layer_details and pool_details
    conv_layers = [input_layer]
    last_layer = input_layer

    for i in range(len(layer_dims) - 1):
        target_dim = layer_dims[i+1]

        new_conv_layer = tf.layers.conv2d(
            inputs=last_layer,
            filters=nfilters,
            kernel_size=[kernel_size, kernel_size],
            padding=padding,
            activation=activation)

        conv_layers.append(new_conv_layer)

        # (kernel_size - 1) from conv layer
        pool_dim = layer_dims[i] - (kernel_size - 1) - (target_dim - 1)

        new_pool = tf.layers.max_pooling2d(inputs=new_conv_layer, pool_size=[pool_dim, pool_dim], strides=stride)

        last_layer = new_pool

    #upsample time!
    for i in range(len(conv_layers) - 2):
        #upsample by nearest neighbor
        corresponding_layer = conv_layers[-(i+1)]

        upsampled_size = corresponding_layer.shape[1:3]

        new_upsample_layer = tf.image.resize_nearest_neighbor(last_layer, size=upsampled_size)
        #add residual layer
        residual_layer = residual_module(corresponding_layer)
        print("upsamp", new_upsample_layer.shape)

        last_layer = tf.add(new_upsample_layer, residual_layer)

    corresponding_layer = conv_layers[0]
    upsampled_size = corresponding_layer.shape[1:3]
    last_layer = tf.image.resize_nearest_neighbor(last_layer, size=upsampled_size)

    #finally, 200 1x1 convolutional layers and we're done
    new_conv_layer = tf.layers.conv2d(
        inputs=last_layer,
        filters=output_size,
        kernel_size=[1,1],
        activation=None)
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
