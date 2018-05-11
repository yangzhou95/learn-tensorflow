import tensorflow as tf

# initialize variables/model parameters
W= tf.Variable(tf.zeros([2, 1]),name='weights')
b=tf.Variable(0.,name='bias')

def inference(X):
    return tf.matmul(X,W)+b

def loss(X,Y):
    Y_predicted=tf.transpose(inference(X)) # make it a row vector
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))

def inputs():
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44],
                  [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34],
                  [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308,
                         220, 311, 181, 274, 303, 244]
    return tf.to_float(weight_age),tf.to_float(blood_fat_content)

def train(total_loss):
    learning_rate=0.000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess,X,Y):
    print("evaluation:", sess.run(inference([[80.,25.]])))

init=tf.global_variables_initializer()

with tf.Session() as sess:
    # sess.run(init)
    tf.global_variables_initializer().run()

    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord=tf.train.Coordinator()
    # starts all queue runners collected in the graph, return a list of threads
    threads = tf.train.start_queue_runners(sess= sess, coord= coord)

    training_step=10000
    for step in range(training_step):
        #sess.run(fetches, feed_dict=None, options=None, run_metadata=None)
        #he `fetches` argument may be a single graph element, or an arbitrarily
        # nested list, tuple, namedtuple, dict, or OrderedDict containing graph
        #elements at its leaves.
        sess.run([train_op])
        if step % 100 == 0:
            print(sess.run([total_loss]))

    print("Final model W=", sess.run(W), "b=", sess.run(b))
    evaluate(sess,X, Y)

    # requests that the threads stop
    coord.request_stop()
    # wait for threads to terminate
    coord.join(threads) # threads: the started threads to join in addition to the registered threads.
    sess.close()












































































