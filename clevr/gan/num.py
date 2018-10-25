import os
import scipy.misc
import numpy as np
import time
from num_model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

import sys

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_string("model_name", "WGAN-GP", "model used: WGAN-GP")
flags.DEFINE_string("base_name", "CNN", "base model used: CNN")

flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 10000000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")

flags.DEFINE_string("data_str", "../dataset/clevr-dataset-gen/dataset/", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("dataset", "blue_cube.blue_sphere.red_cube", "The name of dataset [celebA, mnist, lsun]")

flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("classifier_checkpoint_dir", "classifier_checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("test_sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")

flags.DEFINE_float("gp_coef", 12., "Learning rate of for adam [0.0002]")
flags.DEFINE_integer("wdf_iter", 50, "train discriminator for wdf_iter times in the first f_iter iterations")
flags.DEFINE_integer("f_iter", 50, "see the first one")
flags.DEFINE_integer("d_iter", 5, "discriminator iterations")
flags.DEFINE_integer("g_iter", 1, "generator iterations")

flags.DEFINE_boolean("decay", True, "True for visualizing, False for nothing [False]")

FLAGS = flags.FLAGS

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    FLAGS.sample_dir = 'clevr_samples/%s/%s.%s/%s'%(FLAGS.dataset,FLAGS.model_name,FLAGS.base_name,parse_time())
    FLAGS.test_sample_dir = 'clevr_samples/%s/%s.%s/%s/test_sample'%(FLAGS.dataset,FLAGS.model_name,FLAGS.base_name,parse_time())
    FLAGS.checkpoint_dir = 'clevr_checkpoint/%s/%s.%s/%s'%(FLAGS.dataset,FLAGS.model_name,FLAGS.base_name,parse_time())
    print (FLAGS.sample_dir, FLAGS.checkpoint_dir)
            
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.test_sample_dir):
        os.makedirs(FLAGS.test_sample_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    fsock_err = open('./%s/error.log'%(FLAGS.sample_dir), 'w')               
    fsock_out = open('./%s/out.log'%(FLAGS.sample_dir), 'w')               
    sys.stderr = fsock_err     
    sys.stdout = fsock_out
    
    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            dataset_name=FLAGS.dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            crop=FLAGS.crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
            test_sample_dir=FLAGS.test_sample_dir,
            model_name=FLAGS.model_name,
            base_name=FLAGS.base_name,
            data_str = FLAGS.data_str,
            gp_coef = FLAGS.gp_coef,
            d_iter = FLAGS.d_iter,
            g_iter = FLAGS.g_iter,
            f_iter = FLAGS.f_iter,
            wdf_iter = FLAGS.wdf_iter,
            decay=FLAGS.decay)

        # show_all_variables()
        with open('%s/hyper.log'%FLAGS.sample_dir, 'w') as f_hyper:
            print ("Sample %d*%d*3"%(FLAGS.output_height, FLAGS.output_width), file=f_hyper)
            print ("Model name: %s"%(FLAGS.model_name), file=f_hyper)
            print ("Base name: %s"%(FLAGS.base_name), file=f_hyper)
            print ("Learning rate: %s"%(FLAGS.learning_rate), file=f_hyper)
            print ("Decay: %s"%(FLAGS.decay), file=f_hyper)
            print ("Batch size: %s"%(FLAGS.batch_size), file=f_hyper)
            print ("Discriminator: %s"%(FLAGS.d_iter), file=f_hyper)
            print ("Generator: %s"%(FLAGS.g_iter), file=f_hyper)
            print ("GP-coef: %s"%(FLAGS.gp_coef), file=f_hyper)
            print ("Epoch: %s"%(FLAGS.epoch), file=f_hyper)

        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")
            dcgan.generate_samples(FLAGS)



if __name__ == '__main__':
    tf.app.run()
