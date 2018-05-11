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





















