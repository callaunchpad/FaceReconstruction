import tensorflow as tf
import tensorflow.contrib.slim as slim

'''
Residual Block from figure 3
https://arxiv.org/pdf/1603.06937.pdf
'''
def resBlock(x,channels=256,kernel_size=3, activation=tf.nn.leaky_relu):
    result = tf.layers.conv2d(inputs=x, filters=channels/2, kernel_size=[1,1], activation=activation, padding="same")
    result = tf.layers.conv2d(inputs=result, filters=channels/2, kernel_size=[kernel_size, kernel_size], activation=activation, padding="same")
    result = tf.layers.conv2d(inputs=result, filters=channels, kernel_size=[1,1], activation=activation, padding="same")
    return x + result

'''
Input
    input_layer: A (batch_size, width, height, 3) tensor

    layer_dims: A list of layer dimensions. This assumes that
                the input and all slices of the hourglass are square
Output
    A (batch_size, width, heigh, nfilters) tensor representing the
    output of the hourglass
'''
def get_hourglass(input_layer, layer_dims, output_size=200):
    kernel_size = 3
    nfilters = 256
    activation = tf.nn.leaky_relu
    padding = "same"
    stride = 1
    residual_module = resBlock

    # Using 1x1 convolutions to raise the dimension. Original paper used 7x7 w/ pooling for dimensionality reduction
    # We (probably) don't have the same constraints so this will suffice
    input_layer = tf.layers.conv2d(inputs=input_layer, kernel_size=[1,1], filters=nfilters, padding=padding, activation=activation)


    # Bottom up portion, downsizes by max pooling
    residual_layers = []

    last_layer = input_layer
    for i in range(len(layer_dims)):
        target_dim = layer_dims[i]

        residual_layer = residual_module(input_layer, channels=nfilters, kernel_size=kernel_size, activation=activation)
        residual_layers.append(residual_layer)

        last_layer = tf.layers.conv2d(inputs=last_layer, filters=nfilters, kernel_size=[kernel_size, kernel_size], padding=padding, activation=activation)
        pool_dim = last_layer.shape[1] - (target_dim - 1)
        last_layer = tf.layers.max_pooling2d(inputs=last_layer, pool_size=[pool_dim, pool_dim], strides=stride)

    # Top down, upsampling with residual modules
    while residual_layers:
        residual_layer = residual_layers.pop()
        target_dim = [residual_layer.shape[1], residual_layer.shape[2]]
        upsample_layer = tf.image.resize_nearest_neighbor(images=last_layer, size=target_dim)
        last_layer = tf.add(residual_layer, upsample_layer)

    # 2 final 1x1 layers at the end of the network
    last_layer = tf.layers.conv2d(inputs=last_layer, filters=nfilters, kernel_size=[1,1], activation=activation)
    last_layer = tf.layers.conv2d(inputs=last_layer, filters=output_size, kernel_size=[1,1], activation=None)

    return last_layer

#given an input size and output size, find the right kernel size for CNN
def get_kernel_size(layer_in, layer_out, padding="none"):
    if padding=="none":
        return (layer_in[0]-layer_out[0]+1, layer_in[1]-layer_out[1]+1)

def get_layer_size(layer_in, kernel, padding="none"):
    if padding == "none":
        return (layer_in[0]-kernel[0]+1,layer_in[1]-kernel[1]+1,layer_in[2]-kernel[2]+1)
