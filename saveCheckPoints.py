# Saving training checkpoints

# there is the tf.train.Saver class to save the graph variables in proprietary binary
# files. We should periodically save variables, create a 'checkpoint' file, and
# eventually restore the training from the most recent checkpoint if needed.

# In order to use the Saver we need to slightly change the training loop scaffolding code:

import tensorflow as tf
# create a saver
saver = tf.train.Saver()

# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

    # model setup
    # actual training loop
    for step in range(training_step):
        sess.run([training_op])

        if step % 1000 == 0:
            saver.save(sess, save_path='my-model',global_step=step)
            # create checkpoint files with teh name template my-model-{step}
            # these files store the current values of each variable.
            # by default, the saver keep only the most recent 5 files and delete the rest.

    # evaluation

    saver.save(sess=sess, save_path='my-model',global_step=training_step)

    sess.close()

# recover the training from a certain point
# tf.train.get_checkpoint_state: method, verify if we already have a checkpoint saved
# tf.trian.Saver.restore: method, recover teh variable values

with tf.Session() as sess:
    # model setup
    initial_step=0
    # verify if we don't have a checkpoint saved already
    # os.path.dirname(filename) + os.path.basename(filename) == filename
    # To get the dirname of the absolute path, use
    # os.path.dirname(os.path.abspath(__file__))

    # os.path.dirname(__file__): return the path for current running python script
    # but it will return error if it is executed in commandline

    ckpt=tf.train.get_checkpoint_state(checkpoint_dir=__file__)
    # ckpt.model_checkpoint_path： the location where model is stored
    if ckpt and ckpt.model_checkpoint_path:
        # restores from checkpoint
        saver.restore(sess=sess,save_path=ckpt.model_checkpoint_path)
        # S.rsplit([sep=None][,count=S.count(sep)]): split starting from the end
        # separator will be remove from the returned lists.
        initial_step=int(ckpt.model_checkpoint_path.rsplit('-',1)[1])
    # actual training loop
    for step in range(initial_step,training_steps):
        pass
