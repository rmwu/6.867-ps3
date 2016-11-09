import os
import sys
import math
import time
import argparse

from six.moves import cPickle as pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


DATA_PATH = 'art_data/'
IMAGE_SIZE = 50
NUM_CHANNELS = 3
NUM_LABELS = 11
INCLUDE_TEST_SET = False

class ArtistConvNet:
    def __init__(self, invariance,
                 num_training_steps,
                 dropout_frac,
                 weight_penalty,
                 filter_sizes,
                 depths,
                 pooling_params,
                 training_data_file,
                 plot_progress):
        '''
        Initialize the class by loading the required datasets
        and building the graph.
        '''
        self.load_pickled_dataset(training_data_file)
        self.invariance = invariance
        self.num_training_steps = num_training_steps
        self.dropout_frac = dropout_frac
        self.weight_penalty = weight_penalty
        self.filter_sizes = filter_sizes
        self.depths = depths
        self.pooling_params = pooling_params
        self.plot_progress = plot_progress
        if invariance:
            self.load_invariance_datasets()
        self.graph = tf.Graph()
        self.define_tensorflow_graph()

    def define_tensorflow_graph(self):
        print('\nDefining model...')

        # Hyperparameters
        batch_size = 10
        learning_rate = 0.01
        layer1_filter_size = self.filter_sizes[0]
        layer1_depth = self.depths[0]
        layer1_stride = 2
        layer2_filter_size = self.filter_sizes[1]
        layer2_depth = self.depths[1]
        layer2_stride = 2
        layer3_num_hidden = 64
        layer4_num_hidden = 64
        num_training_steps = self.num_training_steps

        # Add max pooling
        pooling = self.pooling_params["pooling"]
        layer1_pool_filter_size = self.pooling_params["filter_size"]
        layer1_pool_stride = self.pooling_params["stride"]
        layer2_pool_filter_size = self.pooling_params["filter_size"]
        layer2_pool_stride = self.pooling_params["stride"]

        # Enable dropout and weight decay normalization
        dropout_prob = self.dropout_frac # set to < 1.0 to apply dropout, 1.0 to remove
        weight_penalty = self.weight_penalty # set to > 0.0 to apply weight penalty, 0.0 to remove

        with self.graph.as_default():
            # Input data
            tf_train_dataset = tf.placeholder(
                tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, NUM_LABELS))
            tf_valid_dataset = tf.constant(self.val_X)
            tf_test_dataset = tf.placeholder(
                tf.float32, shape=[len(self.val_X), IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

            # Implement dropout
            dropout_keep_prob = tf.placeholder(tf.float32)

            # Network weights/parameters that will be learned
            layer1_weights = tf.Variable(tf.truncated_normal(
                [layer1_filter_size, layer1_filter_size, NUM_CHANNELS, layer1_depth], stddev=0.1))
            layer1_biases = tf.Variable(tf.zeros([layer1_depth]))
            layer1_feat_map_size = int(math.ceil(float(IMAGE_SIZE) / layer1_stride))
            if pooling:
                layer1_feat_map_size = int(math.ceil(float(layer1_feat_map_size) / layer1_pool_stride))

            layer2_weights = tf.Variable(tf.truncated_normal(
                [layer2_filter_size, layer2_filter_size, layer1_depth, layer2_depth], stddev=0.1))
            layer2_biases = tf.Variable(tf.constant(1.0, shape=[layer2_depth]))
            layer2_feat_map_size = int(math.ceil(float(layer1_feat_map_size) / layer2_stride))
            if pooling:
                layer2_feat_map_size = int(math.ceil(float(layer2_feat_map_size) / layer2_pool_stride))

            layer3_weights = tf.Variable(tf.truncated_normal(
                [layer2_feat_map_size * layer2_feat_map_size * layer2_depth, layer3_num_hidden], stddev=0.1))
            layer3_biases = tf.Variable(tf.constant(1.0, shape=[layer3_num_hidden]))

            layer4_weights = tf.Variable(tf.truncated_normal(
              [layer4_num_hidden, NUM_LABELS], stddev=0.1))
            layer4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

            # Model
            def network_model(data):
                '''Define the actual network architecture'''

                # Layer 1
                hidden = add_conv_layer(data, layer1_weights, layer1_biases, layer1_stride)
                if pooling:
                    hidden = add_pool_layer(hidden, layer1_pool_filter_size, layer1_pool_stride, 1)
                
                # Layer 2
                hidden = add_conv_layer(hidden, layer2_weights, layer2_biases, layer2_stride)
                if pooling:
                    hidden = add_pool_layer(hidden, layer2_pool_filter_size, layer2_pool_stride, 2)
                
                # Layer 3
                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
                hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
                hidden = tf.nn.dropout(hidden, dropout_keep_prob)
                
                # Layer 4
                output = tf.matmul(hidden, layer4_weights) + layer4_biases
                return output

            # Training computation
            logits = network_model(tf_train_dataset)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

            # Add weight decay penalty
            loss = loss + weight_decay_penalty([layer3_weights, layer4_weights], weight_penalty)

            # Optimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

            # Predictions for the training, validation, and test data.
            train_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(network_model(tf_valid_dataset))
            test_prediction = tf.nn.softmax(network_model(tf_test_dataset))

            def train_model(num_steps=num_training_steps):
                '''
                Train the model with minibatches in a TensorFlow session.
                Return the final training minibatch accuracy
                and validation accuracy as a tuple.
                '''
                with tf.Session(graph=self.graph) as session:
                    tf.initialize_all_variables().run()
                    print('Initializing variables...')
                    
                    batch_train_accuracy = 0
                    validation_accuracy = 0
                    train_accuracies = []
                    valid_accuracies = []

                    for step in range(num_steps):
                        offset = (step * batch_size) % (self.train_Y.shape[0] - batch_size)
                        batch_data = self.train_X[offset:(offset + batch_size), :, :, :]
                        batch_labels = self.train_Y[offset:(offset + batch_size), :]
                        
                        # Data to feed into the placeholder variables in the tensorflow graph
                        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, 
                                     dropout_keep_prob: dropout_prob}
                        _, l, predictions = session.run([optimizer, loss, train_prediction],
                                                        feed_dict=feed_dict)

                        if self.plot_progress:
                            val_preds = session.run(valid_prediction, feed_dict={dropout_keep_prob : 1.0})
                            valid_accuracies.append(accuracy(val_preds, self.val_Y))

                        if (step % 100 == 0):
                            val_preds = session.run(valid_prediction, feed_dict={dropout_keep_prob : 1.0})
                            batch_train_accuracy = accuracy(predictions, batch_labels)
                            validation_accuracy = accuracy(val_preds, self.val_Y)
                            print('')
                            print('Batch loss at step {:d}: {:f}'.format(step, l))
                            print('Batch training accuracy: {:.2%}'.format(batch_train_accuracy))
                            print('Validation accuracy: {:.2%}'.format(validation_accuracy))

                    if self.plot_progress:
                        plt.plot(range(num_steps), valid_accuracies, color="cornflowerblue", linewidth=0.5)
                    
                    # This code is for the final question
                    if self.invariance:
                        print("\n Obtaining final results on invariance sets!")
                        sets = [self.val_X, self.translated_val_X, self.bright_val_X, self.dark_val_X,
                                self.high_contrast_val_X, self.low_contrast_val_X, self.flipped_val_X,
                                self.inverted_val_X,]
                        set_names = ['normal validation', 'translated', 'brightened', 'darkened',
                                     'high contrast', 'low contrast', 'flipped', 'inverted']
                        
                        for i in range(len(sets)):
                            preds = session.run(test_prediction,
                                feed_dict={tf_test_dataset: sets[i], dropout_keep_prob : 1.0})
                            print("Accuracy on {} data: {:.2%}".format(set_names[i], accuracy(preds, self.val_Y)))

                            # save final preds to make confusion matrix
                            if i == 0:
                                self.final_val_preds = preds

                    return (batch_train_accuracy, validation_accuracy)
            
            # save train model function so it can be called later
            self.train_model = train_model

    def load_pickled_dataset(self, pickle_file):
        print("Loading datasets...")
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f, encoding='latin1')
            self.train_X = save['train_data']
            self.train_Y = save['train_labels']
            self.val_X = save['val_data']
            self.val_Y = save['val_labels']

            if INCLUDE_TEST_SET:
                self.test_X = save['test_data']
                self.test_Y = save['test_labels']
            del save  # hint to help gc free up memory
        print('Training set', self.train_X.shape, self.train_Y.shape)
        print('Validation set', self.val_X.shape, self.val_Y.shape)
        if INCLUDE_TEST_SET:
            print('Test set', self.test_X.shape, self.test_Y.shape)

    def load_invariance_datasets(self):
        with open(DATA_PATH + 'invariance_art_data.pickle', 'rb') as f:
            save = pickle.load(f, encoding='latin1')
            self.translated_val_X = save['translated_val_data']
            self.flipped_val_X = save['flipped_val_data']
            self.inverted_val_X = save['inverted_val_data']
            self.bright_val_X = save['bright_val_data']
            self.dark_val_X = save['dark_val_data']
            self.high_contrast_val_X = save['high_contrast_val_data']
            self.low_contrast_val_X = save['low_contrast_val_data']
            del save

def add_conv_layer(input_data, weights, biases, stride):
    """
    Construct a convolved version of input_data with given
    filter weights, biases, and stride.
    """
    conv = tf.nn.conv2d(input_data, weights, [1, stride, stride, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

def add_pool_layer(input_data, filter_size, stride, layer_num):
    """
    Construct a pooled version of input_data with given
    pooling filter size and stride.
    layer_num helps set optional name of the operation.
    """
    return tf.nn.max_pool(input_data,
                          ksize=[1, filter_size, filter_size, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME',
                          name='pool{}'.format(layer_num))

def weight_decay_penalty(weights, penalty):
    return penalty * sum([tf.nn.l2_loss(w) for w in weights])

def accuracy(predictions, labels):
  return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train/test a convolutional network on art data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--invariance", action="store_true",
                        help="Test finished model on invariance datasets.")
    parser.add_argument("--dropout", type=float, default=1.0,
                        help="Dropout fraction.")
    parser.add_argument("--training_steps", type=int, default=1501,
                        help="Number of steps to train our conv net.")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of times to train with these parameters.")
    parser.add_argument("--weight_penalty", type=float, default=0.0,
                        help="Regularization parameter; "
                             "coefficient on Frobenius weight matrix norms in loss.")
    parser.add_argument("--augmented", action="store_true",
                        help="Use the augmented dataset.")
    parser.add_argument("--plot_progress",
                        help="Location to save plot of validation error over training. "
                             "Do not include the file extension. "
                             "Don't use with repeat > 1.")

    parser.add_argument("--pooling", action="store_true",
                        help="Turn on pooling.")
    parser.add_argument("--pool_stride", type=int, default=2,
                        help="Pooling stride. Does not turn on pooling.")
    parser.add_argument("--pool_filter_size", type=int, default=2,
                        help="Pooling filter size. Does not turn on pooling.")
    parser.add_argument("--filter_sizes", type=int, nargs=2, default=[5, 5],
                        help="Size of the convolutional layer filters.")
    parser.add_argument("--depths", type=int, nargs=2, default=[16, 16],
                        help="Number of feature maps at each convolutional layer.")

    args = parser.parse_args()

    if args.augmented and args.training_steps <= 6000:
        print("Using <= 6000 training steps on augmented data. "
              "The augmented dataset is four times as large as the regular one, "
              "so you should use proportionally more steps.")

    invariance = args.invariance
    if invariance:
        print("Testing finished model on invariance datasets!")
    
    data_filename = 'augmented_art_data.pickle' if args.augmented else "art_data.pickle"
    data_path = os.path.join(DATA_PATH, data_filename)

    def train_single_conv_net(iteration):
        print("\nTraining model #{}".format(iteration))
        conv_net = ArtistConvNet(invariance=invariance,
                                 num_training_steps=args.training_steps,
                                 dropout_frac=args.dropout,
                                 weight_penalty=args.weight_penalty,
                                 filter_sizes=args.filter_sizes,
                                 depths=args.depths,
                                 pooling_params={"pooling": args.pooling,
                                                 "filter_size": args.pool_filter_size,
                                                 "stride": args.pool_stride},
                                 training_data_file=data_path,
                                 plot_progress=args.plot_progress)
        return conv_net.train_model()

    t1 = time.time()
    accuracies = np.array([train_single_conv_net(i) for i in range(args.repeat)])
    t2 = time.time()
    print("Finished training. Total time taken: {}".format(t2 - t1))

    if args.plot_progress:
        plt.title("Validation accuracy during training")
        plt.xlabel("Training steps")
        plt.ylabel("Validation accuracy")

        # directory might not exist
        fig_dir, fig_name = os.path.split(args.plot_progress)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        # with open(os.path.join(fig_dir, fig_name + "-data.pickle"), 'w') as f:
            # pickle.dump((plt.xdata, plt.ydata), f)

        plt.savefig(args.plot_progress + ".pdf", format="pdf")
        plt.show()

    if args.repeat > 1:
        train_acc_mean, val_acc_mean = np.mean(accuracies, axis=0)
        train_acc_sem, val_acc_sem = scipy.stats.sem(accuracies)
        print("Training accuracies: {}".format(accuracies.T[0]))
        print("Validation accuracies: {}".format(accuracies.T[1]))
        print("Mean training accuracy: {:.2%} +- {:.2%}".format(train_acc_mean, train_acc_sem))
        print("Mean validation accuracy: {:.2%} +- {:.2%}".format(val_acc_mean, val_acc_sem))

    # remind us what we ran!
    print("You ran command: {}".format(" ".join(sys.argv)))
