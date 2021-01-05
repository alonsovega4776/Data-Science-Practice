import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plot

def neuron_layer(X, n_neuron, name, act=None):
    with tf.name_scope(name):                                                       # name scope using name of layer
        n_input = int(X.get_shape()[1])                                             # num. of inputs
        std_dev = 2/np.sqrt(n_input)
        initial = tf.random.truncated_normal((n_input, n_neuron), stddev=std_dev)   # no large weights

        W = tf.Variable(initial, name="weights")                                    # [n_input x n_neuron]
        b = tf.Variable(tf.zeros([n_neuron]), name="biases")                        # bias
        z = tf.matmul(X, W) + b                                                     # compute O/P

        if act == "relu":
            return tf.nn.relu(z)                                                    # max{0, z}
        else:
            return z

#-------------------------------------------_Construction_--------------------------------------------------------------

n_input = 28*28
n_hidden1 = 300
n_hidden2 = 300
n_output = 10

tf.compat.v1.disable_eager_execution()

# Tensors
X = tf.compat.v1.placeholder(tf.float32, shape=(None, n_input), name="X")       # feature tensor handle
y = tf.compat.v1.placeholder(tf.int64, shape=(None), name="y")                  # response tensor handle

# NN
with tf.name_scope("dnn"):
    X_1 = neuron_layer(X, n_hidden1, "hidden1", act="relu")
    X_2 = neuron_layer(X_1, n_hidden2, "hidden2", act="relu")
    log_odds = neuron_layer(X_2, n_output, "outputs")

# Cost Function
with tf.name_scope("loss"):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
                                     (labels=y, logits=log_odds)               # cross entropy vector
    obj_func = tf.reduce_mean(entropy, name="loss")                            # mean cross entropy loss function

# Optimization Algorithm
alpha = 0.95                                                                    # learning rate

with tf.name_scope("train"):
    opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=alpha, name="GradientDescent")
    train_op = opt.minimize(obj_func)

# Evaluation
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(targets=y, predictions=log_odds, k=1)
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))

# Initialize and Saver
initial = tf.compat.v1.global_variables_initializer()
save = tf.compat.v1.train.Saver()

#---------------------------------------------_Execution_---------------------------------------------------------------

# Load
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

n_example = len(y_train)                # number of examples
n_test = len(y_test)                  # number of tests

# Scale
X_train = X_train / 255.0
X_test = X_test / 255.0

# Shuffle
combined = list(zip(X_train, y_train))
np.random.shuffle(combined)
X_train, y_train = zip(*combined)

combined = list(zip(X_test, y_test))
np.random.shuffle(combined)
X_test, y_test = zip(*combined)

# Reshape
X_train = tf.reshape(X_train, [n_example, n_input])
X_test = tf.reshape(X_test, [n_test, n_input])

# Train
n_epoch = 1
batch_size = 50

with tf.compat.v1.Session() as ses:
    initial.run()

    for epoch in range(n_epoch):
        for j in range(batch_size, n_example+batch_size, batch_size):
            X_batch = X_train[j-batch_size:j]
            y_batch = y_train[j-batch_size:j]
            ses.run(train_op, feed_dict={X: X_batch, y: y_batch})

        acc_train = acc.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = acc.eval(feed_dict={X: X_test, y: y_test})

        print(epoch, "Train Accuracy: ", acc_train, "Test Accuracy: ", acc_test)

    save_path = save.save(ses, "./model.ckpt")

