import tensorflow as tf

from hourglass import get_hourglass

def get_model(input, name='hourglass'):
    layer_dims = list(reversed(range(20, 200, 10)))

    hourglass = get_hourglass(input, layer_dims, output_size=256)
    hourglass = get_hourglass(hourglass, layer_dims, output_size=200)
    return tf.identity(hourglass, name=name)

def load_model(file_path="./models/chkpt", sess=None):
    '''
    :param file_path: The saver prefix to load the model from
    :param sess: A tensorflow session to use, if None, tf.Session() will be called
    :return: [input (placeholder), label (placeholder), model (tensor), sigmoid_cross_entropy]
    '''
    if sess is None:
        sess = tf.Session()
    saver = tf.train.import_meta_graph(file_path+".meta")
    saver.restore(sess, file_path)
    graph = tf.get_default_graph()
    input = graph.get_tensor_by_name("input:0")
    labels = graph.get_tensor_by_name("labels:0")
    model = graph.get_tensor_by_name("hourglass:0")
    loss = graph.get_tensor_by_name("cross_entropy_loss:0")
    optimizer = graph.get_tensor_by_name("optimizer")
    return input, labels, model, loss, optimizer