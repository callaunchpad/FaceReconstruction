import tensorflow as tf

'''
Residual Block from figure 3
https://arxiv.org/pdf/1603.06937.pdf
'''
def res_block(x,channels=256,kernel_size=3, activation=tf.nn.leaky_relu):
    result = tf.layers.conv2d(inputs=x, filters=channels/2, kernel_size=[1,1], activation=activation, padding="same")
    result = tf.layers.conv2d(inputs=result, filters=channels/2, kernel_size=[kernel_size, kernel_size], activation=activation, padding="same")
    result = tf.layers.conv2d(inputs=result, filters=channels, kernel_size=[1,1], activation=activation, padding="same")
    return x + result

'''
Based off of: https://arxiv.org/pdf/1603.06937.pdf
Input
    input_layer: A (batch_size, width, height, 3) tensor

    layer_dims: A list of layer dimensions. This assumes that
                the input and all slices of the hourglass are square
Output
    A (batch_size, width, heigh, nfilters) tensor representing the
    output of the hourglass
'''
def hourglass(input_layer, num_layers, output_channels=200, activation=tf.nn.tanh, int_channels=256):

    # Using 1x1 convolutions to raise the dimension. Original paper used 7x7 w/ pooling for dimensionality reduction
    # We (probably) don't have the same constraints so this will suffice
    last_layer = tf.layers.conv2d(inputs=input_layer, kernel_size=[1,1], filters=int_channels, padding="SAME", activation=activation)

    # Maintaining layer shapes due so that upsampling isn't off by one due to rounding
    residual_layers = []
    for _ in range(num_layers):
        '''Unclear whether 1 or 3 conv layers should be used per layer'''
        # for _ in range(3):
        for _ in range(1):
            last_layer = tf.layers.conv2d(last_layer, kernel_size=[3,3], filters=int_channels, padding="SAME", activation=activation)

        #Skip layer occurs before pooling but after convolutions
        residual_layer = res_block(last_layer, channels=int_channels, kernel_size=3, activation=activation)
        residual_layers.append(residual_layer)

        last_layer = tf.layers.max_pooling2d(last_layer, 2, 2)
        pass

    # Upsample and incorporating residuals
    for residual_layer in reversed(residual_layers):
        target_shape = [residual_layer.shape[1], residual_layer.shape[2]]
        upsampled = tf.image.resize_nearest_neighbor(images=last_layer, size=target_shape)
        last_layer = residual_layer + upsampled
        pass


    #Final 2 rounds of 1x1 convolutions
    last_layer = tf.layers.conv2d(inputs=last_layer, kernel_size=[1,1], filters=int_channels, padding="SAME", activation=activation)
    last_layer = tf.layers.conv2d(inputs=last_layer, kernel_size=[1,1], filters=output_channels, padding="SAME", activation=activation)

    return last_layer
