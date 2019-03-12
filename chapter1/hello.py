import tensorflow as tf

b = tf.constant(0.1, shape=[33])
w = tf.truncated_normal([2,3],stddev=0.1)
sess = tf.Session()

print(sess.run(b))
print(sess.run(w))
# 
