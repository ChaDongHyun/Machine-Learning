import tensorflow as tf

# set X and Y (using placeholder)
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# set weight, bias and hypothesis
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = X * W + b

# set cost(loss)function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# make optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start training
for step in range(2001):
    cost_val, w_val, b_val, _ = sess.run([cost, W, b, train], feed_dict = {X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1,
                                                                                                   6.1]})
    if step % 20 == 0 :
        print(step, cost_val, w_val, b_val)

# testing model
print('\n=== Testing ===')
print(sess.run(hypothesis,feed_dict = {X: [5]}))
print(sess.run(hypothesis, feed_dict = {X: [2.5]}))
print(sess.run(hypothesis, feed_dict = {X: [1.5, 3.5]}))