# read csv to model
import tensorflow as tf
import numpy as np
import os


def read_csv(batch_size, file_name, record_defaults=1):
    fileName_queue=tf.train.string_input_producer(os.path.dirname(__file__)+"/"+file_name)
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value=reader.read(fileName_queue,name='read_op')

    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets teh data type for each column
    decoded=tf.decode_csv(records=value)

    # batch actually reads the file and loads "batch size" rows in a single tensor
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size* 50, min_after_dequeue=batch_size)


def inputs():
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked =\
    read_csv(100,"./data/train.csv",)