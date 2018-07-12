from keras.models import load_model
import tensorflow as tf

print(len(tf.trainable_variables()))

model = load_model('model1.h5', compile=False)

print(len(tf.trainable_variables()))
model.summary()
# graph.get_tensor
# print(a)
with tf.Session() as sess:
    # writer = tf.summary.FileWriter("./log", sess.graph)
    graph = tf.get_default_graph()
    a = graph.get_tensor_by_name("x1:0")
    ops = graph.get_operations()
    print(ops)