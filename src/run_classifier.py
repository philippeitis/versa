"""
Script to reproduce the few-shot classification results in:
"Meta-Learning Probabilistic Inference For Prediction"
https://arxiv.org/pdf/1805.09921.pdf

The following command lines will reproduce the published results within error-bars:

Omniglot 5-way, 5-shot
----------------------
python run_classifier.py

Omniglot 5-way, 1-shot
----------------------
python run_classifier.py --shot 1

Omniglot 20-way, 5-shot
-----------------------
python run_classifier.py --way 20 --iterations 60000

Omniglot 20-way, 1-shot
-----------------------
python run_classifier.py --way 20 --shot 1 --iterations 100000

minImageNet 5-way, 5-shot
-------------------------
python run_classifier.py --dataset miniImageNet --tasks_per_batch 4 --iterations 100000 --dropout 0.5

minImageNet 5-way, 1-shot
-------------------------
python run_classifier.py --dataset miniImageNet --shot 1 --tasks_per_batch 8 --iterations 50000 --dropout 0.5 -lr 0.00025

"""

import logging

import numpy as np
import tensorflow as tf

from features import extract_features_omniglot, extract_features_mini_imagenet
from inference import infer_classifier
from utilities import sample_normal, multinoulli_log_density, print_and_log, get_log_files
from data import get_data

"""
parse_command_line: command line parser
"""


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", choices=["Omniglot", "miniImageNet"],
                        default="Omniglot", help="Dataset to use")
    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                        help="Whether to run traing only, testing only, or both training and testing.")
    parser.add_argument("--d_theta", type=int, default=256,
                        help="Size of the feature extractor output.")
    parser.add_argument("--shot", type=int, default=5,
                        help="Number of training examples.")
    parser.add_argument("--way", type=int, default=5,
                        help="Number of classes.")
    parser.add_argument("--test_shot", type=int, default=None,
                        help="Shot to be used at evaluation time. If not specified 'shot' will be used.")
    parser.add_argument("--test_way", type=int, default=None,
                        help="Way to be used at evaluation time. If not specified 'way' will be used.")
    parser.add_argument("--tasks_per_batch", type=int, default=16,
                        help="Number of tasks per batch.")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples from q.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--iterations", type=int, default=80000,
                        help="Number of training iterations.")
    parser.add_argument("--checkpoint_dir", "-c", default='./checkpoint',
                        help="Directory to save trained models.")
    parser.add_argument("--dropout", type=float, default=0.9,
                        help="Dropout keep probability.")
    parser.add_argument("--test_model_path", "-m", default=None,
                        help="Model to load and test.")
    parser.add_argument("--print_freq", type=int, default=200,
                        help="Frequency of summary results (in iterations).")
    args = parser.parse_args()

    # adjust test_shot and test_way if necessary
    if args.test_shot is None:
        args.test_shot = args.shot
    if args.test_way is None:
        args.test_way = args.way

    if args.mode == "test" and not args.test_model_path:
        parser.error("--test was selected but no test model path was provided.")

    return args


def main(_unused_argv):
    logger = tf.get_logger()
    logger.setLevel(logging.ERROR)

    args = parse_command_line()

    logfile, checkpoint_path_validation, checkpoint_path_final = get_log_files(args.checkpoint_dir)

    print_and_log(logfile, "Options: %s\n" % args)

    # Load training and eval data
    data = get_data(args.dataset)

    # set the feature extractor based on the dataset
    if args.dataset == "Omniglot":
        feature_extractor_fn = extract_features_omniglot
    else:
        feature_extractor_fn = extract_features_mini_imagenet

    # evaluation samples
    eval_samples_train = 15
    eval_samples_test = args.shot

    # testing parameters
    test_iterations = 600
    test_args_per_batch = 1  # always use a batch size of 1 for testing

    # tf placeholders
    train_images = tf.keras.Input(
        # shot, *dimensions
        [None, data.get_image_height(), data.get_image_width(), data.get_image_channels()],
        dtype=tf.float32,
        name='train_images'
    )
    test_images = tf.keras.Input(
        # num test images, *dimensions
        [None, data.get_image_height(), data.get_image_width(), data.get_image_channels()],
        dtype=tf.float32,
        name='test_images'
    )
    train_labels = tf.keras.Input(
        # shot, way
        [None, args.way],
        dtype=tf.float32,

        name='train_labels'
    )
    test_labels = tf.keras.Input(
        # num test images, way
        [None, args.way],
        dtype=tf.float32,
        name='test_labels'
    )
    dropout_rate = tf.compat.v1.placeholder(
        tf.float32,
        [],
        name='dropout_rate'
    )
    L = tf.constant(args.samples, dtype=tf.float32, name="num_samples")

    # Relevant computations for a single task
    def evaluate_task(inputs):
        train_inputs, train_outputs, test_inputs, test_outputs = inputs
        with tf.compat.v1.variable_scope('shared_features'):
            # extract features from train and test data
            features_train = feature_extractor_fn(
                images=train_inputs,
                output_size=args.d_theta,
                use_batch_norm=True,
                rate=dropout_rate
            )
            features_test = feature_extractor_fn(
                images=test_inputs,
                output_size=args.d_theta,
                use_batch_norm=True,
                rate=dropout_rate
            )
        # Infer classification layer from q
        with tf.compat.v1.variable_scope('classifier'):
            classifier = infer_classifier(
                features_train, train_outputs, args.d_theta, args.way
            )

        # Local reparameterization trick
        # Compute parameters of q distribution over logits
        weight_mean = classifier['weight_mean']
        weight_log_variance = classifier['weight_log_variance']

        bias_mean = classifier["bias_mean"]
        bias_log_variance = classifier['bias_log_variance']

        logits_mean_test = tf.matmul(features_test, weight_mean) + bias_mean
        logits_log_var_test = \
            tf.math.log(tf.matmul(features_test ** 2, tf.exp(weight_log_variance)) + tf.exp(bias_log_variance))
        logits_sample_test = sample_normal(logits_mean_test, logits_log_var_test, args.samples)

        test_labels_tiled = tf.tile(tf.expand_dims(test_outputs, 0), [args.samples, 1, 1])
        task_log_py = multinoulli_log_density(inputs=test_labels_tiled, logits=logits_sample_test)

        averaged_predictions = tf.reduce_logsumexp(logits_sample_test, axis=0) - tf.math.log(L)

        task_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_outputs, axis=-1),
                                                        tf.argmax(averaged_predictions, axis=-1)), tf.float32))
        task_score = tf.reduce_logsumexp(task_log_py, axis=0) - tf.math.log(L)
        task_loss = -tf.reduce_mean(task_score, axis=0)

        return [task_loss, task_accuracy]

    # tf mapping of batch to evaluation function
    batch_output = tf.map_fn(
        fn=evaluate_task,
        elems=(train_images, train_labels, test_images, test_labels),
        dtype=[tf.float32, tf.float32],
        parallel_iterations=args.tasks_per_batch
    )

    # average all values across batch
    batch_losses, batch_accuracies = batch_output
    loss = tf.reduce_mean(batch_losses)
    accuracy = tf.reduce_mean(batch_accuracies)

    def run_batches(num_iters, mode="test"):
        outputs = []
        if mode == "test":
            batch_args = [mode, test_args_per_batch, args.test_shot, args.test_way, eval_samples_test]
        else:
            batch_args = [mode, args.tasks_per_batch, args.shot, args.way, eval_samples_train]
        for _ in range(num_iters):
            train_inputs, test_inputs, train_outputs, test_outputs = \
                data.get_batch(*batch_args)
            feed_dict = {
                train_images: train_inputs,
                test_images: test_inputs,
                train_labels: train_outputs,
                test_labels: test_outputs,
                dropout_rate: 1. - args.dropout if mode == "train" else 0.
            }
            if mode == "train":
                _, iter_loss, iter_acc = sess.run([train_step, loss, accuracy], feed_dict)
                outputs.append((iter_loss, iter_acc))
            else:
                outputs.append(sess.run([accuracy], feed_dict))
        return outputs

    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.compat.v1.train.Saver()

        if "train" in args.mode:
            # train the model
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)
            train_step = optimizer.minimize(loss)

            validation_batches = 200
            best_validation_accuracy = 0.0
            sess.run(tf.compat.v1.global_variables_initializer())
            # Main training loop
            for train_loop in range(0, args.iterations, args.print_freq):
                train_accuracies = []
                for iter_loss, iter_accuracy in run_batches(args.print_freq, mode="train"):
                    train_accuracies.append(iter_accuracy)

                # compute accuracy on validation set
                validation_accuracies = run_batches(validation_batches, mode="validation")
                validation_accuracy = np.array(validation_accuracies).mean()
                train_accuracy = np.array(train_accuracies).mean()

                # save checkpoint if validation is the best so far
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    saver.save(sess=sess, save_path=checkpoint_path_validation)

                print_and_log(logfile, 'Iteration: {}, Loss: {:5.3f}, Train-Acc: {:5.3f}, Val-Acc: {:5.3f}'
                              .format(train_loop + args.print_freq, iter_loss, train_accuracy, validation_accuracy))
            # save the checkpoint from the final epoch
            saver.save(sess, save_path=checkpoint_path_final)
            print_and_log(logfile, f"Fully-trained model saved to: {checkpoint_path_final}")
            print_and_log(logfile, f"Best validation accuracy: {best_validation_accuracy:5.3f}")
            print_and_log(logfile, f"Best validation model saved to: {checkpoint_path_validation}")

        def test_model(model_path, load=True):
            if load:
                saver.restore(sess, save_path=model_path)
            test_accuracies = run_batches(test_iterations)

            test_accuracy = np.array(test_accuracies).mean() * 100.0
            confidence_interval_95 = (
                    196.0 * np.array(test_accuracies).std() / np.sqrt(len(test_accuracies))
            )
            print_and_log(logfile, 'Held out accuracy: {0:5.3f} +/- {1:5.3f} on {2:}'
                          .format(test_accuracy, confidence_interval_95, model_path))

        if args.mode == 'train_test':
            print_and_log(logfile, 'Train Shot: {0:d}, Train Way: {1:d}, Test Shot {2:d}, Test Way {3:d}'
                          .format(args.shot, args.way, args.test_shot, args.test_way))
            # test the model on the final trained model
            # no need to load the model, it was just trained
            test_model(checkpoint_path_final, load=False)

            # test the model on the best validation checkpoint so far
            test_model(checkpoint_path_validation)
        elif args.mode == 'test':
            test_model(args.test_model_path)

    logfile.close()


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.app.run()
