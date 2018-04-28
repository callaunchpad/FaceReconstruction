import tensorflow as tf
import numpy as np

sess = tf.Session()

loader = tf.train.import_meta_graph('./models/chkpt.meta')
loader.restore(sess, tf.train.latest_checkpoint('./models/'))

graph = tf.get_default_graph()
w = graph.get_tensor_by_name("w:0")
b = graph.get_tensor_by_name("b:0")

w, b = sess.run([w, b])
print(w, b)

sess.close()