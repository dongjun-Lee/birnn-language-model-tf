import tensorflow as tf
from tensorflow.contrib import rnn


class RNNLanguageModel(object):
    def __init__(self, vocabulary_size, args):
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_layers
        self.num_hidden = args.num_hidden

        self.x = tf.placeholder(tf.int32, [None, None])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.shape(self.x)[0]

        self.lm_input = self.x[:, :-1]
        self.lm_output = self.x[:, 1:]
        self.seq_len = tf.reduce_sum(tf.sign(self.lm_input), 1)

        with tf.name_scope("embedding"):
            init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            lm_input_emb = tf.nn.embedding_lookup(embeddings, self.lm_input)

        with tf.variable_scope("rnn"):
            def make_cell():
                cell = rnn.BasicLSTMCell(self.num_hidden)
                cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                return cell
            cell = rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])
            rnn_outputs, _ = tf.nn.dynamic_rnn(
                cell, lm_input_emb, sequence_length=self.seq_len, dtype=tf.float32)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(rnn_outputs, vocabulary_size)

        with tf.name_scope("loss"):
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits,
                targets=self.lm_output,
                weights=tf.sequence_mask(self.seq_len, tf.shape(self.x)[1] - 1, dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=True
            )
