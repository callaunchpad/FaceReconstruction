import tensorflow as tf
import numpy as np

w = tf.Variable(.1, tf.float32, name="w") #or w = tf.Variable([.1], tf.float32)
b = tf.Variable(.2, tf.float32, name="b")
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

lin_model = w*x + b
loss = tf.reduce_mean(tf.square(lin_model - y))

expW = 5
expB = -4
simple_fxn = lambda x: expW * x + expB

x_train = np.arange(20)
y_train = simple_fxn(x_train)
data = {x: x_train, y: y_train}

optimizer = tf.train.AdamOptimizer(0.005)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()

for i in range(10000):
    sess.run(train, data)
    if(i % 100 == 0): 
        print(i, sess.run(loss, data))
        saver.save(sess, './models/chkpt')


w, b = sess.run([w, b])
saver.save(sess, './models/chkpt')
print(w, b)

sess.close()