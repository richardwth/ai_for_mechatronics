import sys
sys.path.insert(0, '/home/richard_wth/ai_mecha/')
from GeneralTools.misc_fun import FLAGS
FLAGS.DEFAULT_IN = '/data/cephfs/punim0811/Datasets/cifar10'
FLAGS.DEFAULT_DOWNLOAD = '/data/cephfs/punim0811/'
FLAGS.DEFAULT_OUT = '/home/richard_wth/ai_mecha/Results/cifar'
####################################################################
from GeneralTools.my_input import ReadTFRecords
import tensorflow as tf
import os.path

with tf.Graph().as_default():
    # get batch of images
    dataset = ReadTFRecords(
        'cifar_xy', num_features=3*32*32, num_labels=1,
        x_dtype=tf.string, y_dtype=tf.int64, batch_size=8)
    dataset.shape2image(3, 32, 32)
    data_batch = dataset.next_batch()
    x_batch = data_batch['x']
    # change [channel, height width] to [height, width, channel]
    x_batch = tf.transpose(x_batch, perm=(0, 2, 3, 1))

    # add to summary
    tf.summary.image('cifar', x_batch, max_outputs=8)
    summary_op = tf.summary.merge_all()
    summary_folder = os.path.join(FLAGS.DEFAULT_OUT, 'cifar_log')
    if not os.path.exists(summary_folder):
        os.makedirs(summary_folder)

    # run the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(summary_folder, sess.graph)
        xv, summary_str = sess.run([x_batch, summary_op])
        summary_writer.add_summary(summary_str, global_step=0)

    print('x_batch shape {}'.format(xv.shape))
    print('Job finished successfully.')


