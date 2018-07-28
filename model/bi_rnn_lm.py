import tensorflow as tf
from tensorflow.contrib import rnn


class BiRNNLanguageModel(object):
    def __init__(self, vocabulary_size, args):
        self.embedding_size = args.embedding_size
        self.num_layers = args.num_layers
        self.num_hidden = args.num_hidden

        self.x = tf.placeholder(tf.int32, [None, None])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.shape(self.x)[0]

        self.lm_input = self.x
        self.lm_output = self.x[:, 1:-1]
        self.seq_len = tf.reduce_sum(tf.sign(self.lm_input), 1)

        with tf.name_scope("embedding"):
            init_embeddings = tf.random_uniform([vocabulary_size, self.embedding_size])
            embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            lm_input_emb = tf.nn.embedding_lookup(embeddings, self.lm_input)

        with tf.name_scope("bi-rnn"):
            def make_cell():
                cell = rnn.BasicLSTMCell(self.num_hidden)
                cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                return cell

            fw_cell = rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])
            bw_cell = rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, lm_input_emb, sequence_length=self.seq_len, dtype=tf.float32)

            fw_outputs = rnn_outputs[0][:, :-2, :]
            bw_outputs = rnn_outputs[1][:, 2:, :]
            merged_output = tf.concat([fw_outputs, bw_outputs], axis=2)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(merged_output, vocabulary_size)

        with tf.name_scope("loss"):
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits,
                targets=self.lm_output,
                weights=tf.sequence_mask(self.seq_len - 2, tf.shape(self.x)[1] - 2, dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=True
            )
