import tensorflow as tf
# set x_data, y_data
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# set X and Y placeholder, set W variable
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([1]), name='weight')

# make simplified hypothesis and cost function
hypothesis = X * W
cost = tf.reduce_sum(tf.square(hypothesis - Y))

''' 
gradient descent part
(same)
optimizer  = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = train.minimize(cost) 
'''
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# initialize variable
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start training
for step in range(21) :
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))