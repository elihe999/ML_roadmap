import tensorflow as tf

mystr = tf.Variable("111", tf.string) 
cool_number = tf.Variable([3.1415, 2.71828], tf.float32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.32j], tf.complex64)
print(mystr)

mymat = tf.Variable([[7],[11]], tf.int16)
squarish_squares = tf.Variable([[4, 9], [16, 25]], tf.int32)
print(tf.rank(mymat))
print(tf.rank(squarish_squares))