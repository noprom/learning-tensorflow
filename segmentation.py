# -*- coding:utf-8 -*-
import tensorflow as tf

sess = tf.InteractiveSession()
seg_ids = tf.constant([0, 1, 1, 2, 2])
tens1 = tf.constant([[2, 5, 3, -5],
                     [0, 3, -2, 5],
                     [4, 3, 5, 3],
                     [6, 1, 4, 0],
                     [6, 1, 4, 0]])
sum_seg = tf.segment_sum(tens1, seg_ids).eval()  # Sum segmentation
prod_seg = tf.segment_prod(tens1, seg_ids).eval()  # Product segmentation
min_seg = tf.segment_min(tens1, seg_ids).eval()  # minimun value goes to group
max_seg = tf.segment_max(tens1, seg_ids).eval()  # maximum value goes to group
mean_seg = tf.segment_mean(tens1, seg_ids).eval()  # mean value goes to group
