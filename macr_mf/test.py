import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.python.ops.gen_math_ops import neg
from tensorflow.python.ops.special_math_ops import _exponential_space_einsum


initializer = tf.contrib.layers.xavier_initializer()
neg_sample=2
batch_size=5
embed_size=3
neg_items = tf.Variable(initializer([batch_size*neg_sample,embed_size]))
pos_items = tf.Variable(initializer([batch_size,embed_size]))
usr = tf.Variable(initializer([batch_size,embed_size]))
neg_items_pop = tf.Variable(initializer([batch_size*neg_sample,embed_size]))
pos_items_pop = tf.Variable(initializer([batch_size,embed_size]))
usr_pop = tf.Variable(initializer([batch_size,embed_size]))



tiled_usr=tf.reshape(tf.tile(usr,[1,neg_sample]),[-1,embed_size])
tiled_usr_pop=tf.reshape(tf.tile(usr_pop,[1,neg_sample]),[-1,embed_size])
pos_item_score=tf.reduce_sum(tf.multiply(usr,pos_items),axis=1)
neg_item_score=tf.reduce_sum(tf.multiply(tiled_usr,neg_items),axis=1)
pos_item_pop_score=tf.reduce_sum(tf.multiply(usr_pop,pos_items_pop),axis=1)
neg_item_pop_score=tf.reduce_sum(tf.multiply(tiled_usr_pop,neg_items_pop),axis=1)

neg_item_pop_score_exp=tf.reduce_sum(tf.exp(tf.reshape(neg_item_pop_score,[-1,neg_sample])),axis=1)
pos_item_pop_score_exp=tf.exp(pos_item_score)
loss2=tf.reduce_mean(tf.negative(tf.log(pos_item_pop_score_exp/(pos_item_pop_score_exp+neg_item_pop_score_exp))))

weighted_pos_item_score=tf.multiply(pos_item_score,tf.sigmoid(pos_item_pop_score))
weighted_neg_item_score=tf.multiply(neg_item_pop_score,tf.sigmoid(neg_item_pop_score))
neg_item_score_exp=tf.reduce_sum(tf.exp(tf.reshape(weighted_neg_item_score,[batch_size,-1])),axis=1)
pos_item_score_exp=tf.exp(weighted_pos_item_score)
loss1=tf.reduce_mean(tf.negative(tf.log(pos_item_score_exp/(pos_item_score_exp+neg_item_score_exp))))







with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(usr))
    print(sess.run(tiled_usr))
    print(sess.run(3*tf.sigmoid(pos_item_pop_score)))
    print(sess.run(loss1))
    print(sess.run(loss2))