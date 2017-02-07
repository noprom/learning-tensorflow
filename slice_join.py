# -*- coding:utf-8 -*-
import tensorflow as tf

sess = tf.InteractiveSession()

t_matrix = tf.constant([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
t_array = tf.constant([1, 2, 3, 4, 9, 8, 6, 5])
t_array2 = tf.constant([2, 3, 4, 5, 6, 7, 8, 9])
tf.slice(t_matrix, [1, 1], [2, 2]).eval()  # cutting an slice
tf.split(0, 2, t_array)  # splitting the array in two
tf.tile([1, 2], [3]).eval()  # tiling this little tensor 3 times
tf.pad(t_matrix, [[0, 1], [2, 1]]).eval()  # padding
tf.concat(0, [t_array, t_array2]).eval()  # concatenating list
tf.pack([t_array, t_array2]).eval()  # packing
sess.run(tf.unpack(t_matrix))  # Unpacking, we need the run method to view the tensors
tf.reverse(t_matrix, [False, True]).eval()  # Reverse matrix
