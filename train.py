import argparse
import numpy as np
import tensorflow as tf
from data_utils import build_word_dict, build_dataset, batch_iter
from model.rnn_lm import RNNLanguageModel
from model.bi_rnn_lm import BiRNNLanguageModel


def train(train_data, test_data, vocabulary_size, args):
    with tf.Session() as sess:
        if args.model == "rnn":
            model = RNNLanguageModel(vocabulary_size, args)
        elif args.model == "birnn":
            model = BiRNNLanguageModel(vocabulary_size, args)
        else:
            raise ValueError("Unknown model option {}.".format(args.model))

        # Define training procedure
        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

        # Summary
        loss_summary = tf.summary.scalar("loss", model.loss)
        summary_op = tf.summary.merge([loss_summary])
        train_summary_writer = tf.summary.FileWriter(args.model + "-train", sess.graph)
        test_summary_writer = tf.summary.FileWriter(args.model + "-test", sess.graph)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(batch_x):
            feed_dict = {model.x: batch_x, model.keep_prob: args.keep_prob}
            _, step, summaries, loss = sess.run([train_op, global_step, summary_op, model.loss], feed_dict=feed_dict)
            train_summary_writer.add_summary(summaries, step)

            if step % 100 == 1:
                print("step {0}: loss = {1}".format(step, loss))

        def test_perplexity(test_data, step):
            test_batches = batch_iter(test_data, args.batch_size, 1)
            losses, iters = 0, 0

            for test_batch_x in test_batches:
                feed_dict = {model.x: test_batch_x, model.keep_prob: 1.0}
                summaries, loss = sess.run([summary_op, model.loss], feed_dict=feed_dict)
                test_summary_writer.add_summary(summaries, step)
                losses += loss
                iters += 1

            return np.exp(losses / iters)

        batches = batch_iter(train_data, args.batch_size, args.num_epochs)
        for batch_x in batches:
            train_step(batch_x)
            step = tf.train.global_step(sess, global_step)

            if step % 100 == 1:
                perplexity = test_perplexity(test_data, step)
                print("\ttest perplexity: {}".format(perplexity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="birnn", help="rnn | birnn")
    parser.add_argument("--embedding_size", type=int, default=300, help="embedding size.")
    parser.add_argument("--num_layers", type=int, default=1, help="RNN network depth.")
    parser.add_argument("--num_hidden", type=int, default=300, help="RNN network size.")
    parser.add_argument("--keep_prob", type=float, default=0.5, help="dropout keep prob.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate.")

    parser.add_argument("--batch_size", type=int, default=64, help="batch size.")
    parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs.")
    args = parser.parse_args()

    train_file = "ptb_data/ptb.train.txt"
    test_file = "ptb_data/ptb.test.txt"
    word_dict = build_word_dict(train_file)
    train_data = build_dataset(train_file, word_dict)
    test_data = build_dataset(test_file, word_dict)

    train(train_data, test_data, len(word_dict), args)
