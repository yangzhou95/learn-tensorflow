# 1.An inference model is a series of mathmatical operations that we apply
# to our data.

# 2. The arbitrary values inside these fixed operations are the parameters
# of the model, and are the ones that change through training in order for
# the model to learn and adjust its output

# 3. We always apply the same general structure for training inference models
# no matter how significant variation of the number of operations they use

# Training loop:
# 1.Initialize the model parameters with random values or zeros (for the first time)
# 2.Reads the training data along with the expected output(Y) for each data example (x)
# 3.Executes the inference model on the training data. For each input example(x)
#   calculate its output (Y_predicted) with current model parameters
# 4.Computes the loss, which is the distance between expected output(Y) and
#   predicted output(Y_predicted).
# 5.Adjust model parameters.
# 6.After training, evaluate the model. Execute the inference against a different dataset to which
#   we also have the expected output.


# Scaffolding for the model

import tensorflow as tf
import numpy as np

# 1. Initialize Variable/model parameters

# 2. Define the training loop operations

def inputs():
    # read/generate input training data X and expected outputs Y.

def inference(X):
    # compute the inference model over data x and return the Y_predicted.

def loss(X,Y):
    # compute loss over training data X and expected value Y.

def train(total_loss):
    # train/adjust model parameters according to computed total loss

def evaluate(sess, X, Y):
    # evaluate the resulting trained model

# 3. Launch the graph in a session, setup boilerplate
with tf.Session() as sess:
    tf.global_variables_initializer().run() # All the variables must be initialized
                                            # before using them
    X, Y=input()
    total_loss=loss(X,Y)
    train_op= train(total_loss)

    # main thread
    coord=tf.train.Coordinator()
    #create new subthread
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)

    # Actual training loop
    training_steps=1000
    for step in range(training_steps):
        sess.run([train_op])
        if step % 10 == 0:
            print("loss: ",sess.run([total_loss]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads=threads)
    sess.close()










