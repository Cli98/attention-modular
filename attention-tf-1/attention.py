from __future__ import division
from __future__ import print_function

from datetime import datetime
from data import data_iterator_len
from data import read_vocabulary

import tensorflow as tf
import numpy as np
import sys
import os
from bleu.length_analysis import process_files


def get_bleu_score(samples, target_file):
    import time
    hyp_file = "hyp" + str(int(time.time()))
    if not os.path.isdir("./predictions_Mon"):
        os.makedirs("./predictions_Mon")
    with open("./predictions_Mon/" + hyp_file, "w") as f:
        for sample in samples:
            for s in sample:
                if s == "</s>": break
                f.write(" " + s)
            f.write("\n")

    process_files("./predictions_Mon/"+hyp_file, target_file)
    # os.remove(hyp_file)

class AttentionNN(object):
    def __init__(self, config, sess):
        self.sess          = sess
        self.hidden_size   = config.hidden_size
        self.num_layers    = config.num_layers
        self.batch_size    = config.batch_size
        self.max_size      = config.max_size
        self.init_dropout  = config.dropout
        self.epochs        = config.epochs
        self.s_nwords      = config.s_nwords
        self.t_nwords      = config.t_nwords
        self.minval        = config.minval
        self.maxval        = config.maxval
        self.lr_init       = config.lr_init
        self.max_grad_norm = config.max_grad_norm
        self.dataset       = config.dataset
        self.emb_size      = config.emb_size
        self.is_test       = config.is_test
        self.name          = config.name

        self.source_data_path  = config.source_data_path
        self.target_data_path  = config.target_data_path
        self.source_vocab_path = config.source_vocab_path
        self.target_vocab_path = config.target_vocab_path
        self.checkpoint_dir    = config.checkpoint_dir

        self.train_iters = 0

        if not os.path.isdir(self.checkpoint_dir):
            print("[!] Directory {} not found".format(self.checkpoint_dir))
            print("make a new dir")
            os.makedirs(self.checkpoint_dir)

        self.source     = tf.placeholder(tf.int32, [self.batch_size, self.max_size], name="source")
        self.target     = tf.placeholder(tf.int32, [self.batch_size, self.max_size], name="target")
        self.target_len = tf.placeholder(tf.int32, [self.batch_size], name="target_len")
        self.dropout    = tf.placeholder(tf.float32, name="dropout")

        self.build_variables()
        self.build_model()

    def build_variables(self):
        #self.lr = tf.Variable(self.lr_init, trainable=False, name="lr")
        initializer = tf.random_uniform_initializer(self.minval, self.maxval)

        with tf.variable_scope("encoder"):
            self.s_emb = tf.get_variable("s_embedding", shape=[self.s_nwords, self.emb_size],
                                         initializer=initializer)
            self.s_proj_W = tf.get_variable("s_proj_W", shape=[self.emb_size, self.hidden_size],
                                            initializer=initializer)
            self.s_proj_b = tf.get_variable("s_proj_b", shape=[self.hidden_size],
                                            initializer=initializer)
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.dropout))
            self.encoder = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

        with tf.variable_scope("decoder"):
            self.t_emb = tf.get_variable("t_embedding", shape=[self.t_nwords, self.emb_size],
                                         initializer=initializer)
            self.t_proj_W = tf.get_variable("t_proj_W", shape=[self.emb_size, self.hidden_size],
                                            initializer=initializer)
            self.t_proj_b = tf.get_variable("t_proj_b", shape=[self.hidden_size],
                                            initializer=initializer)
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=(1-self.dropout))
            self.decoder = tf.nn.rnn_cell.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

            # projection
            self.proj_W = tf.get_variable("W", shape=[self.hidden_size, self.emb_size],
                                          initializer=initializer)
            self.proj_b = tf.get_variable("b", shape=[self.emb_size],
                                          initializer=initializer)
            self.proj_Wo = tf.get_variable("Wo", shape=[self.emb_size, self.t_nwords],
                                           initializer=initializer)
            self.proj_bo = tf.get_variable("bo", shape=[self.t_nwords],
                                           initializer=initializer)

            # attention
            self.v_a = tf.get_variable("v_a", shape=[self.hidden_size, 1],
                                       initializer=initializer)
            self.W_a = tf.get_variable("W_a", shape=[2*self.hidden_size, self.hidden_size],
                                       initializer=initializer)
            self.b_a = tf.get_variable("b_a", shape=[self.hidden_size],
                                       initializer=initializer)
            self.W_c = tf.get_variable("W_c", shape=[2*self.hidden_size, self.hidden_size],
                                       initializer=initializer)
            self.b_c = tf.get_variable("b_c", shape=[self.hidden_size],
                                       initializer=initializer)

    def build_model(self):
        with tf.variable_scope("encoder"):
            source_xs = tf.nn.embedding_lookup(self.s_emb, self.source)
            source_xs = tf.split(source_xs, self.max_size, 1)
        with tf.variable_scope("decoder"):
            target_xs = tf.nn.embedding_lookup(self.t_emb, self.target)
            target_xs = tf.split(target_xs, self.max_size, 1)

        s = self.encoder.zero_state(self.batch_size, tf.float32)
        encoder_hs = []
        with tf.variable_scope("encoder"):
            for t in range(self.max_size):
                if t > 0: tf.get_variable_scope().reuse_variables()
                x = tf.squeeze(source_xs[t], [1])
                x = tf.matmul(x, self.s_proj_W) + self.s_proj_b
                h, s = self.encoder(x, s)
                encoder_hs.append(h)
        encoder_hs = tf.unstack(encoder_hs)

        s = self.decoder.zero_state(self.batch_size, tf.float32)
        logits = []
        probs  = []
        with tf.variable_scope("decoder"):
            for t in range(self.max_size):
                if t > 0: tf.get_variable_scope().reuse_variables()
                if not self.is_test or t == 0:
                    x = tf.squeeze(target_xs[t], [1])
                x = tf.matmul(x, self.t_proj_W) + self.t_proj_b
                h_t, s = self.decoder(x, s)
                h_tld = self.attention(h_t, encoder_hs)

                oemb  = tf.matmul(h_tld, self.proj_W) + self.proj_b
                logit = tf.matmul(oemb, self.proj_Wo) + self.proj_bo
                prob  = tf.nn.softmax(logit)
                logits.append(logit)
                probs.append(prob)
                if self.is_test:
                    x = tf.cast(tf.argmax(prob, 1), tf.int32)
                    x = tf.nn.embedding_lookup(self.t_emb, x)

        logits     = logits[:-1]
        targets    = tf.split( self.target, self.max_size, 1)[1:]
        weights    = tf.unstack(tf.sequence_mask(self.target_len - 1, self.max_size - 1,
                                                dtype=tf.float32), None, 1)
        self.loss  = tf.contrib.legacy_seq2seq.sequence_loss(logits, targets, weights)
        self.probs = tf.transpose(tf.stack(probs), [1, 0, 2])
        self.optimizer = tf.train.GradientDescentOptimizer(1)
        self.optim = tf.contrib.layers.optimize_loss(self.loss, None,
                self.lr_init, self.optimizer, clip_gradients=5.,
                summaries=["learning_rate", "loss", "gradient_norm"])

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()

    def attention(self, h_t, encoder_hs):
        #scores = [tf.matmul(tf.tanh(tf.matmul(tf.concat(1, [h_t, tf.squeeze(h_s, [0])]),
        #                    self.W_a) + self.b_a), self.v_a)
        #          for h_s in tf.split(0, self.max_size, encoder_hs)]
        #scores = tf.squeeze(tf.pack(scores), [2])
        scores = tf.reduce_sum(tf.multiply(encoder_hs, h_t), 2)
        a_t    = tf.nn.softmax(tf.transpose(scores))
        a_t    = tf.expand_dims(a_t, 2)
        c_t    = tf.matmul(tf.transpose(encoder_hs, perm=[1,2,0]), a_t)
        c_t    = tf.squeeze(c_t, [2])
        h_tld  = tf.tanh(tf.matmul(tf.concat([h_t, c_t], 1), self.W_c) + self.b_c)

        return h_tld

    def get_model_name(self):
        date = datetime.now()
        return "{}-{}-{}-{}-{}".format(self.name, self.dataset, date.month, date.day, date.hour)

    def train(self, epoch, merged_sum, writer):
        #if epoch > 10 and epoch % 5 == 0 and self.lr_init > 0.00025:
        #    self.lr_init = self.lr_init*0.75
        #    self.lr.assign(self.lr_init).eval()
        if epoch > 8:
            self.optimizer._learning_rate /= 2.0
        #print(self.optimizer._learning_rate)
        total_loss = 0.
        i = 0
        iterator = data_iterator_len(self.source_data_path,
                                     self.target_data_path,
                                     read_vocabulary(self.source_vocab_path),
                                     read_vocabulary(self.target_vocab_path),
                                     self.max_size, self.batch_size,limit=None)
        for dsource, slen, dtarget, tlen in iterator:
            outputs = self.sess.run([self.loss, self.optim, merged_sum],
                                    feed_dict={self.source: dsource,
                                               self.target: dtarget,
                                               self.target_len: tlen,
                                               self.dropout: self.init_dropout})
            loss = outputs[0]
            # itr  = self.train_iters*epoch + i
            itr = i
            total_loss += loss
            if itr % 2 == 0:
                writer.add_summary(outputs[-1], itr)
            if itr % 10 == 0:
                print("[Train] [Time: {}] [Epoch: {}] [Iteration: {}] [lr: {}] [Loss: {}] [Perplexity: {}]"
                      .format(datetime.now(), epoch, itr, self.optimizer._learning_rate, loss, np.exp(loss)))
                sys.stdout.flush()
            i += 1
        self.train_iters = i
        return total_loss/i

    def test(self, source_data_path, target_data_path):
        iterator = data_iterator_len(source_data_path,
                                     target_data_path,
                                     read_vocabulary(self.source_vocab_path),
                                     read_vocabulary(self.target_vocab_path),
                                     self.max_size, self.batch_size)

        total_loss = 0
        i = 0
        for dsource, slen, dtarget, tlen in iterator:
            loss, = self.sess.run([self.loss],
                                  feed_dict={self.source: dsource,
                                             self.target: dtarget,
                                             self.target_len: tlen,
                                             self.dropout: 0.0})
            total_loss += loss
            i += 1
            if i%100==0:
                print(str(i),"items have been processed!")

        total_loss /= i
        return total_loss

    def sample(self, source_data_path):
        source_vocab = read_vocabulary(self.source_vocab_path)
        target_vocab = read_vocabulary(self.target_vocab_path)
        inv_target_vocab = {target_vocab[k]:k for k in target_vocab}
        iterator = data_iterator_len(source_data_path,
                                     source_data_path,
                                     source_vocab,
                                     target_vocab,
                                     self.max_size, self.batch_size)
        samples = []
        for dsource, slen, dtarget, tlen in iterator:
            dtarget = [[target_vocab["<s>"]] + [target_vocab["<pad>"]]*(self.max_size-1)]
            dtarget = dtarget*self.batch_size
            probs, = self.sess.run([self.probs],
                                   feed_dict={self.source: dsource,
                                              self.target: dtarget,
                                              self.dropout: 0.0})
            for b in range(self.batch_size):
                samples.append([inv_target_vocab[np.argmax(p)] for p in probs[b]])

        return samples

    def sample_in_place(self, source_data_path, target_data_path):
        source_vocab = read_vocabulary(self.source_vocab_path)
        target_vocab = read_vocabulary(self.target_vocab_path)
        inv_target_vocab = {target_vocab[k]:k for k in target_vocab}
        iterator = data_iterator_len(source_data_path,
                                     target_data_path,
                                     source_vocab,
                                     target_vocab,
                                     self.max_size, self.batch_size)
        samples = []
        for dsource, slen, dtarget, tlen in iterator:
            #dtarget = [[target_vocab["<s>"]] + [target_vocab["<pad>"]]*(self.max_size-1)]
            #dtarget = dtarget*self.batch_size
            probs, = self.sess.run([self.probs],
                                   feed_dict={self.source: dsource,
                                              self.target: dtarget,
                                              self.dropout: 0.0})
            for b in range(self.batch_size):
                samples.append([inv_target_vocab[np.argmax(p)] for p in probs[b]])

        return samples

    def run(self, valid_source_data_path, valid_target_data_path):
        merged_sum = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/{}".format(self.get_model_name()),
                                        self.sess.graph)

        best_valid_loss = float("inf")
        for epoch in range(7,self.epochs):
            train_loss = self.train(epoch, merged_sum, writer)
            self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.name + ".epoch" + str(epoch)))
            valid_loss = self.test(valid_source_data_path, valid_target_data_path)
            print("[Train] [Avg. Loss: {}] [Avg. Perplexity: {}]".format(train_loss, np.exp(train_loss)))
            print("[Valid] [Loss: {}] [Perplexity: {}]".format(valid_loss, np.exp(valid_loss)))

            samples = self.sample_in_place(valid_source_data_path, valid_target_data_path)
            get_bleu_score(samples, valid_target_data_path)
            if epoch == 0 or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.name + ".bestvalid"))

    def load(self):
        print("[*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        #print("Current path: ", os.getcwd())
        #print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            if os.path.exists("./checkpoints/default.epoch6.meta"):
                print("Find a new ckpt!")
                print("Checkpoint path: ", ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, "./checkpoints/default.epoch6")
                # self.sess()
            else:
                print("[!] No checkpoint found, start to train new model!")

        else:
            print("[!] No checkpoint found, start to train new model!")



