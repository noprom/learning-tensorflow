# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

trainsamples = 200
testsamples = 60


# Here we will represent the model, a simple imput, a hidden layer of sigmoid activation
def model(X, hidden_weights1, hidden_bias1, ow):
    hidden_layer = tf.nn.sigmoid(tf.matmul(X, hidden_weights1) + hidden_bias1)
    return tf.matmul(hidden_layer, ow)


dsX = np.linspace(-1, 1, trainsamples + testsamples).transpose()
dsY = 0.4 * pow(dsX, 2) + 2 * dsX + np.random.randn(*dsX.shape) * 0.22 + 0.8
plt.figure()  # Create a new figure
plt.title('Original data')
plt.scatter(dsX, dsY)  # Plot a scatter draw of the datapoints
# plt.show()

X = tf.placeholder("float")
Y = tf.placeholder("float")
# Create first hidden layer
hw1 = tf.Variable(tf.random_normal([1, 10], stddev=0.1))  # Create output connection
ow = tf.Variable(tf.random_normal([10, 1], stddev=0.0))  # Create bias
b = tf.Variable(tf.random_normal([10], stddev=0.1))

model_y = model(X, hw1, b, ow)
# Cost function
cost = tf.pow(model_y - Y, 2) / (2)
# construct an optimizer
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)  # Launch the graph in a session
with tf.Session() as sess:
    tf.global_variables_initializer().run()  # Initialize all variables
    for i in range(1, 100):
        dsX, dsY = shuffle(dsX.transpose(), dsY)  # We randomize the samples to mplement a better training
        trainX, trainY = dsX[0:trainsamples], dsY[0:trainsamples]
        for x1, y1 in zip(trainX, trainY):
            sess.run(train_op, feed_dict={X: [[x1]], Y: y1})
        testX, testY = dsX[trainsamples:trainsamples + testsamples], dsY[0:trainsamples:trainsamples + testsamples]
        cost1 = 0.
        for x1, y1 in zip(testX, testY):
            cost1 += sess.run(cost, feed_dict={X: [[x1]], Y: y1}) / testsamples
        if (i % 10 == 0):
            print "Average cost for epoch " + str(i) + ":" + str(cost1)