__author__ = 'alange'
"""
Predict input image
"""

import numpy as np
import theano
import theano.tensor as T

import lasagne

from PIL import Image




class Predictor:
    """
    Class for managing network and prediction
    """

    def __init__(self):
        self.network = self.load_network()
        # self.prediction = lasagne.layers.get_output(self.network, deterministic=True)

    @staticmethod
    def load_network():
        input_var = T.tensor4('inputs')
        network = Predictor.build_cnn(input_var)
        read_filename = "static/model_baconnet_color.npz"
        print("Loading parameters from previous run from file: {}".format(read_filename))
        loaded_params = np.load(read_filename)
        lasagne.layers.set_all_param_values(network, loaded_params['arr_0'])
        return network

    @staticmethod
    def build_cnn(input_var=None):
        # ##################### Build the neural network model #######################
        network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64), input_var=input_var)
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

    def predict(self, image_data):
        """
        Given JPEG image data, convert to 3 channel matrix/tensor input for network
        """
        input = Predictor.process_input_image(Predictor.write_jpeg_to_disk(image_data))
        prediction = lasagne.layers.get_output(self.network, inputs=input, deterministic=True)
        prediction = prediction.eval()[0]  # Evaluate the Theano object
        return {"not": prediction[0], "bacon": prediction[1], "kevin": prediction[2]}

    @staticmethod
    def write_jpeg_to_disk(image_data):
        """
        Write the file to disk for easy reading by PIL later
        """
        filename = "tmp.jpg"
        with open(filename, "wb") as outfile:
            outfile.write(image_data)
        return filename

    @staticmethod
    def process_input_image(filename):
        """
        Read JPEG file, convert image into 64x64 matrix
        """
        im = Image.open(filename)
        im = im.resize((64, 64))
        r_vec = []
        g_vec = []
        b_vec = []
        for i in range(0, 64):
            for j in range(0, 64):
                r, g, b = im.getpixel((i, j))
                r_vec.append(r)
                g_vec.append(g)
                b_vec.append(b)
        npa = np.asarray(r_vec + g_vec + b_vec, dtype=np.float64)
        npa = npa.reshape((-1, 3, 64, 64))
        return npa
