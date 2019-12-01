from __future__ import division
from __future__ import print_function

from attention import AttentionNN
from data import read_vocabulary


import os
import time
import pprint
import random
import numpy as np
import tensorflow as tf

tf.reset_default_graph()

pp = pprint.PrettyPrinter().pprint

flags = tf.app.flags

flags.DEFINE_integer("max_size", 50, "Maximum sentence length [30]")
flags.DEFINE_integer("batch_size", 128, "Number of examples in minibatch [128]")
flags.DEFINE_integer("random_seed", 123, "Value of random seed [123]")
flags.DEFINE_integer("epochs", 12, "Number of epochs to run [10]")
flags.DEFINE_integer("hidden_size", 1024, "Size of hidden units [1024]")
flags.DEFINE_integer("emb_size", 256, "Size of embedding dimension [256]")
flags.DEFINE_integer("num_layers", 4, "Depth of RNNs [4]")
flags.DEFINE_float("dropout", 0.2, "Dropout probability [0.2]")
flags.DEFINE_float("minval", -0.1, "Minimum value for initialization [-0.1]")
flags.DEFINE_float("maxval", 0.1, "Maximum value for initialization [0.1]")
flags.DEFINE_float("lr_init", 1.0, "Initial learning rate [1.0]")
flags.DEFINE_float("max_grad_norm", 5.0, "Maximum gradient cutoff [5.0]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Checkpoint directory [checkpoints]")
flags.DEFINE_string("dataset", "medium", "Dataset to use [small]")
flags.DEFINE_string("name", "default", "Model name [default]")
flags.DEFINE_string("sample", None, "Sample from dataset [None]")
flags.DEFINE_string("source_data_path", None, "Sample from dataset [None]")
flags.DEFINE_string("target_data_path", None, "Sample from dataset [None]")
flags.DEFINE_string("valid_source_data_path", None, "Sample from dataset [None]")
flags.DEFINE_string("valid_target_data_path", None, "Sample from dataset [None]")
flags.DEFINE_string("test_source_data_path", None, "Sample from dataset [None]")
flags.DEFINE_string("test_target_data_path", None, "Sample from dataset [None]")
flags.DEFINE_string("source_vocab_path", None, "Sample from dataset [None]")
flags.DEFINE_string("target_vocab_path", None, "Sample from dataset [None]")
flags.DEFINE_string("s_nwords", None, "Sample from dataset [None]")
flags.DEFINE_string("t_nwords", None, "Sample from dataset [None]")
flags.DEFINE_string("checkpoint", "checkpoints_sample_test", "Sample from dataset [None]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for training [False]")

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)


class debug:
    source_data_path       = "data/train.debug.en"
    target_data_path       = "data/train.debug.vi"
    source_vocab_path      = "data/vocab.small.en"
    target_vocab_path      = "data/vocab.small.vi"
    valid_source_data_path = "data/test.debug.en"
    valid_target_data_path = "data/test.debug.vi"
    test_source_data_path  = "data/test.debug.en"
    test_target_data_path  = "data/test.debug.vi"


class small:
    source_data_path       = "data/train.small.en.pruned"
    target_data_path       = "data/train.small.vi.pruned"
    source_vocab_path      = "data/vocab.small.en"
    target_vocab_path      = "data/vocab.small.vi"
    valid_source_data_path = "data/valid.small.en.pruned"
    valid_target_data_path = "data/valid.small.vi.pruned"
    test_source_data_path  = "data/tst2013.en.pruned"
    test_target_data_path  = "data/tst2013.vi.pruned"


class medium:
    source_data_path  = "data/train.medium.en"
    target_data_path  = "data/train.medium.de"
    source_vocab_path = "data/vocab.medium.en"
    target_vocab_path = "data/vocab.medium.de"
    valid_source_data_path = "data/newstest2014.en"
    valid_target_data_path = "data/newstest2014.de"
    test_source_data_path  = "data/newstest2014.en"
    test_target_data_path  = "data/newstest2014.de"


def print_samples(samples):
    for sample in samples:
        for s in sample:
            if s == "</s>":
                break
            print(" " + s, end="")
        print()


def main(_):
    config = FLAGS
    if config.dataset == "small":
        data_config = small
    elif config.dataset == "medium":
        data_config = medium
    elif config.dataset == "debug":
        data_config = debug
    else:
        raise Exception("[!] Unknown dataset {}".format(config.dataset))
    #print(data_config.source_data_path)
    config.source_data_path  = data_config.source_data_path
    config.target_data_path  = data_config.target_data_path
    config.source_vocab_path = data_config.source_vocab_path
    config.target_vocab_path = data_config.target_vocab_path

    s_nwords = len(read_vocabulary(config.source_vocab_path))
    t_nwords = len(read_vocabulary(config.target_vocab_path))

    config.s_nwords = s_nwords
    config.t_nwords = t_nwords
    #print("config:", config.__dict__)
    #print("end")
    #pp(config.__dict__["__flags"])
    gpu_options = tf.GPUOptions(visible_device_list="0")
    # ckpt_name = "default.epoch0"
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # saver = tf.train.import_meta_graph(os.path.join("checkpoints",ckpt_name+".meta"))
        # saver.restore(sess, os.path.join("checkpoints",ckpt_name))
        attn = AttentionNN(config, sess)
        if config.sample:
            attn.load()
            samples = attn.sample(config.sample)
            print_samples(samples)
        else:
            if not config.is_test:
                attn.load()
                print("start training!")
                attn.run(data_config.valid_source_data_path,
                         data_config.valid_target_data_path)
            else:
                attn.load()
                loss = attn.test(data_config.test_source_data_path,
                                 data_config.test_target_data_path)
                print("[Test] [Loss: {}] [Perplexity: {}]".format(loss, np.exp(loss)))
                samples = attn.sample_in_place(data_config.test_source_data_path, data_config.test_target_data_path)
                from attention import get_bleu_score
                get_bleu_score(samples, data_config.test_target_data_path)


if __name__ == "__main__":

    tf.app.run()
