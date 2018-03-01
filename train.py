# A straight-forward CNN for fitting the MNIST dataset
# Gets about 95% accurate. Meh.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import mnist_parse
import sys
#%matplotlib inline

tf.set_random_seed(42)

X_train,y_train = mnist_parse.parse("train")
X_train = X_train.reshape(60000,28,28,1)/255.

X_test,y_test = mnist_parse.parse("test")
X_test = X_test.reshape(10000,28,28,1)/255.


nbatches=60
X_train_batches = []
y_train_batches = []
for i in range(0,60):
    X_train_batches.append(X_train[i*60:(i+1)*60,...])
    y_train_batches.append(y_train[i*60:(i+1)*60,...])

#w1_dummy = np.array([  [[1,0,-1],[1,0,-1],[1,0,-1] ],[[1,1,1],[0,0,0],[-1,-1,-1] ]],dtype=np.float32).reshape(3,3,1,2)
#W1 = tf.Variable(w1_dummy)

X = tf.placeholder(shape=(None,28,28,1),dtype=tf.float32)
y = tf.placeholder(shape=(None,10),dtype=tf.float32)

# Layer 1 : 3x3x1x10 convolution

W1 = tf.Variable(tf.random_normal((3,3,1,10), stddev=0.1))
B1 = tf.Variable(tf.zeros((1,1,1,10)))

Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME') + B1
A1 = tf.nn.relu(Z1)

# Layer 2 : 3x3x10x10 convolution

W2 = tf.Variable(tf.random_normal((3,3,10,10), stddev=0.1))
B2 = tf.Variable(tf.zeros((1,1,1,10)))

Z2 = tf.nn.conv2d(A1,W2, strides = [1,1,1,1], padding = 'SAME') + B2
A2 = tf.nn.relu(Z2)

# Layer 2b : max pooling

Z3 = tf.nn.max_pool(A2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
print(Z3.shape)
# Layer 3 :

W4 = tf.Variable(tf.random_normal((3,3,10,20), stddev=0.1))
B4 = tf.Variable(tf.zeros((1,1,1,20)))

Z4 = tf.nn.conv2d(Z3,W4, strides = [1,1,1,1], padding = 'SAME') +B4
A4 = tf.nn.relu(Z4)

Z5 = tf.nn.max_pool(A4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W6 = tf.Variable(tf.random_normal((3,3,20,20), stddev=0.1))
B6 = tf.Variable(tf.zeros((1,1,1,20)))

Z6 = tf.nn.conv2d(Z5,W6, strides = [1,1,1,1], padding = 'SAME') +B6
A6 = tf.nn.relu(Z6)

Z7 = tf.nn.max_pool(A6,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

Z7_flat = tf.contrib.layers.flatten(Z7)


W8 = tf.Variable(tf.random_normal((320,10)))
B8 = tf.Variable(tf.zeros((1,10)))

Z8 = tf.matmul(Z7_flat,W8) + B8

prediction = tf.argmax(Z8,1)
print(prediction.shape)

cost_accuracy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=Z8))
cost_l2 = tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(W4)+tf.nn.l2_loss(W6)+tf.nn.l2_loss(W8)
cost = cost_accuracy+0.05 * cost_l2

X_sample = X_train_batches[0]
y_sample = y_train_batches[0]

updates = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

stepno = 0
steps = []
costs = []
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(0,30):
        for batch in range(nbatches):
            X_sample = X_train_batches[batch]
            y_sample = y_train_batches[batch]

            _,costnum,pred = sess.run([updates,cost,prediction],feed_dict={X:X_sample,y:y_sample})
            frac_right = np.sum(pred == np.argmax(y_sample,1))/y_sample.shape[0]

            print(epoch,batch,costnum,frac_right)
            steps.append(stepno)
            stepno += 1
            costs.append(costnum)


            #print(pred)
            #print(np.argmax(y_sample,1))
    costnum,pred = sess.run([cost,prediction],feed_dict={X:X_test,y:y_test})
    print("test cost:",costnum)
    print(pred.shape,y_test.shape,np.argmax(y_test,1).shape)
    print("test frac right:",np.sum(pred == np.argmax(y_test,1))/y_test.shape[0]  )

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(steps,costs)
plt.savefig("cost.png")
