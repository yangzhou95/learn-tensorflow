<<<<<<< HEAD
# There is a function used commonly in machine learning called the logistic
#  function. Itis also known as the sigmoid function, because its shape is an S
# (and sigma is the greekletter equivalent to s).

# The logistic function is a probability distribution function that, given a specific input
# value, computes the probability of the output being a success, and thus the probability for
# the answer to the question to be “yes.”

# Logistic regression example in Tensorflow using Kaggle's Titanic Dataset

import tensorflow as tf
import  numpy as np
import os

# 1. initialize parameters
W=tf.Variable(tf.zeros(5,1),name='weights')
b=tf.Variable(0., name='bias')

# sum up the inputs that modified by weights and biases
def combine_input(X):
    return tf.matmul(X,W)+b

# the infered valules is the outputs of the sigmoid function
def inference(X)
    return tf.sigmoid(combine_input(X))

# define loss function
def loss(X,Y)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(combine_input(X),Y))

def read_csv(batch_size, file_name, record_defaults):
    filename_queue=tf.train.string_input_producer([os.path.dirname(__file__)+'/'+file_name])

    reader=tf.TextLineReader(skip_header_lines=1)
    key, value=reader.read(filename_queue)

    # decoded_csv will convert a tensor from type string (the text line) in (into)
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column. Each column is a tensor
    decoded = tf.decode_csv(value)














=======
import tensorflow as tf
import os

W=tf.Variable(tf.zeros([5,1]),name='weight')
b=tf.Variable(0.,name='bias')


def combine_inputs(X):
    return tf.matmul(X,W)+b

def inference(X):
    return tf.sigmoid(combine_inputs(X))

def loss(X,Y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(combine_inputs(X),Y))

def read_csv(batch_size, file_name, record_defaults):
        filename_queue = tf.train.string_input_producer([os.path.join(os.getcwd(),file_name)])
        reader=tf.TextLineReader(skip_header_lines=1)
        key, value = reader.read(filename_queue)

        # decode_csv will convert a Tensor from type string( the text line) in
        # a tuple of tensor columns with the specified defaults, which also
        # sets the data type for each column
        decoded = tf.decode_csv(records=value,record_defaults=record_defaults)

        # batch actually reads teh file and loads 'batch_size' rows in a single tensor
        return tf.train.shuffle_batch(tensors=decoded, batch_size=batch_size, capacity=batch_size* 20, min_after_dequeue=batch_size)

def inputs():
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked= \
    read_csv(batch_size=100,file_name="train.csv", record_defaults=
        [[0.0],[0.0],[0],[""],[""],[0.0],[0.0],[0.0],[""],[0.0],[""],[""]])

    # convert categorial data
    is_first_class =tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))

    gender = tf.to_float(tf.equal(sex, ["female"]))

    # Finally, we stack all the features in a single matrix
    # we then transpose to have a matrix with one example per row and one feature per column
    features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))

    survived =tf.reshape(survived,[100,1])

    return features, survived

def train(total_loss):
    learning_rate=0.0001
    return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
    predicted = tf.cast(inference(X)>0.5, tf.float32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted,Y), tf.float32))))

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        if step % 100 == 0:
            print("loss:", sess.run([total_loss]))

    evaluate(sess, X, Y)

    import time
    time.sleep(5)

    coord.request_stop()
    coord.join(threads)
    sess.close()
>>>>>>> cbcff71d39d73fd20584ea5dd40fecd22f011d88







