import tensorflow as tf


tf.debugging.set_log_device_placement(True)
tf.compat.v1.enable_eager_execution()

a = tf.constant([1.])
b = tf.constant([2.])
c = tf.add(a, b)
print(c)

with tf.device('/GPU:0'):

  a = tf.constant([1.0, 2.0, 3.0])

  b = tf.constant([4.0, 5.0, 6.0])



c = tf.add(a, b)

print(c)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), tf.config.list_physical_devices('GPU'))