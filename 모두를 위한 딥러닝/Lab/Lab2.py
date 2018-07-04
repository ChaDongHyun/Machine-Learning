import tensorflow as tf

# set X, Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Set weight, bias and hypothesis
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = x_train * W + b

# set cost(loss) function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# make optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start train
for step in range(2001):
    sess.run(train)
    if step % 20 == 0 :
        print(step, sess.run(cost), sess.run(W), sess.run(b))
