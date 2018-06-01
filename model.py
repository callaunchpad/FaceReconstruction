import tensorflow as tf

from hourglass2 import hourglass

def get_model(input, name='hourglass', depth=3):
    model = hourglass(input, depth, output_channels=256)
    model = hourglass(model, depth, output_channels=200)
    return tf.identity(model, name=name)

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
    step_size = graph.get_tensor_by_name("step_size:0")
    optimizer = graph.get_operation_by_name("optimizer")
    return input, labels, model, loss, step_size, optimizer
