#!/usr/bin/env python

"""
BaconNet: Color

Convolution Neural Net for bacon detection

Like BaconNet, but uses smaller image and RGB channels.
Intended to provide auxiliary information to BaconNet for ensemble result

Based on Lasagne MNIST tutorial
"""

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


def shuffle_in_unison_inplace(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

# ################## Load the bacon dataset ##################
def load_dataset():

    # Read in data
    X_train = np.fromfile("generated_data64/X_train.dat", dtype=np.float32)
    X_val = np.fromfile("generated_data64/X_val.dat", dtype=np.float32)
    X_test = np.fromfile("generated_data64/X_test.dat", dtype=np.float32)

    X_train = X_train.reshape((-1, 3, 64, 64))
    X_val = X_val.reshape((-1, 3, 64, 64))
    X_test = X_test.reshape((-1, 3, 64, 64))

    X_train = X_train.astype(np.float64)
    X_val = X_val.astype(np.float64)
    X_test = X_test.astype(np.float64)

    y_train = np.fromfile("generated_data64/y_train.dat", dtype=np.uint8)
    y_val = np.fromfile("generated_data64/y_val.dat", dtype=np.uint8)
    y_test = np.fromfile("generated_data64/y_test.dat", dtype=np.uint8)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ##################### Build the neural network model #######################
def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(16, 16),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.2),
            num_units=512,
            nonlinearity=lasagne.nonlinearities.sigmoid)

    network = lasagne.layers.DenseLayer(network,
            num_units=3,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs=3, read_params=False, read_filename="model_baconnet_color.npz"):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    print("Train size: {}".format(len(X_train)))
    print("Validation size: {}".format(len(X_val)))
    print("Test size: {}".format(len(X_test)))

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    print("Building network...")
    network = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.sum()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Read parameters from previous fit if requested
    if read_params:
        print("Loading parameters from previous run from file: {}".format(read_filename))
        loaded_params = np.load(read_filename)
        lasagne.layers.set_all_param_values(network, loaded_params['arr_0'])

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.01, momentum=0.9)
    updates = lasagne.updates.rmsprop(loss, params, learning_rate=0.0005, rho=0.9)
    # updates = lasagne.updates.sgd(loss, params, learning_rate=0.0001)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.sum()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")

    batchsize = 200

    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batchsize, shuffle=True):
            inputs, targets = batch
            batch_err = train_fn(inputs, targets)
            pp = lasagne.layers.get_output(network, inputs)
            train_err += batch_err
            print("Batch: {} Loss: {}".format(train_batches, batch_err))
            sys.stdout.flush()
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, len(X_val), shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        # Checkpoint save every epoch
        if epoch % 1 == 0:
            np.savez('epoch_baconnet_color' + str(epoch) + '.npz', lasagne.layers.get_all_param_values(network))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batchsize, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))


    # Optionally, you could now dump the network weights to a file like this:
    np.savez('model_baconnet_color.npz', lasagne.layers.get_all_param_values(network))


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on bacon data using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_epochs'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['read_params'] = sys.argv[2] == "True"
        if len(sys.argv) > 3:
            kwargs['read_filename'] = sys.argv[3]
        main(**kwargs)
