# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# add to kfkd.py
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
import theano.tensor as T
from nolearn.lasagne import BatchIterator
from theano.sandbox.neighbours import neibs2images
from lasagne.objectives import squared_error

### this is really dumb, current nolearn doesnt play well with lasagne,
### so had to manually copy the file I wanted to this folder
from shape import ReshapeLayer

from lasagne.nonlinearities import tanh
import pickle
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score
import os
from IPython.display import Image as IPImage
from PIL import Image


# <codecell>

class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):
        """
        :param incoming: NOT UNDERSTOOD
        :param shape: shape of outgoing array
        :param kwargs:
        :return:
        """
        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            raise ValueError('ds must have len == 2')
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must have len == 2')
            if ds[0] != ds[1]:
                raise ValueError('ds should be symmetric (I am lazy)')
            self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(2, axis=2).repeat(2, axis=3)


# <codecell>

### when we load the batches to input to the neural network, we randomly / flip
### rotate the images, to artificially increase the size of the training set

class FlipBatchIterator(BatchIterator):

    def transform(self, X1, X2):
        X1b, X2b = super(FlipBatchIterator, self).transform(X1, X2)
        X2b = X2b.reshape(X1b.shape)

        bs = X1b.shape[0]
        h_indices = np.random.choice(bs, bs / 2, replace=False)  # horizontal flip
        v_indices = np.random.choice(bs, bs / 2, replace=False)  # vertical flip

        ###  uncomment these lines if you want to include rotations (images must be square)  ###
        #r_indices = np.random.choice(bs, bs / 2, replace=False) # 90 degree rotation
        for X in (X1b, X2b):
            X[h_indices] = X[h_indices, :, :, ::-1]
            X[v_indices] = X[v_indices, :, ::-1, :]
            #X[r_indices] = np.swapaxes(X[r_indices, :, :, :], 2, 3)
        shape = X2b.shape
        X2b = X2b.reshape((shape[0], -1))

        return X1b, X2b

# <codecell>
def loaddata():
    import netCDF4 as nc
    import glob

    xtrain = np.empty(shape=(12540 - 2 * 1254, 80, 80))
    xval = np.empty(shape=(1250, 80, 80))
    xtest = np.empty(shape=(1250, 80, 80))
    trainindex = 0
    valindex = 0
    testindex = 0
    marchingindex = 0

    for filename in glob.glob("*.nc"):
        data = nc.Dataset(filename, 'r', format="NETCDF4")
        sst = data.variables['sst']  # This is a masked array see:
                                     # http://docs.scipy.org/doc/numpy/reference/maskedarray.generic.html
        for day in sst:
            if marchingindex == 9:
                xval[valindex] = day.data[80*4:100*4, 220*4:240*4]
                valindex += 1
            elif marchingindex == 8:
                xtest[testindex] = day.data[80*4:100*4, 220*4:240*4]
                testindex += 1
            else:
                xtrain[trainindex] = day.data[80*4:100*4, 220*4:240*4]
                trainindex += 1
            marchingindex += 1
            if marchingindex == 10:
                marchingindex = 0

    ytrain = xtrain.reshape((xtrain.shape[0], -1))
    yval = xval.reshape((xval.shape[0], -1))
    ytest = xtest.reshape((xtest.shape[0], -1))

    return xtrain, ytrain, xval, yval, xtest, ytest


def get_output_from_nn(last_layer, X):
    indices = np.arange(128, X.shape[0], 128)
    sys.stdout.flush()

    # not splitting into batches can cause a memory error
    x_batches = np.split(X, indices)
    out = []
    for count, X_batch in enumerate(x_batches):
        out.append(layers.helper.get_output(last_layer, X_batch).eval())
        sys.stdout.flush()
    return np.vstack(out)


def decode_encoded_input(final_layer, X):
    return get_output_from_nn(final_layer, X)


def encode_input(encode_layer, X):
    return get_output_from_nn(encode_layer, X)


def main():
    xtrain, ytrain, xval, yval, xtest, ytest = loaddata()

    # <codecell>
    conv_filters = 32
    deconv_filters = 32
    filter_sizes = 7
    epochs = 20
    encode_size = 40
    ae = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv', layers.Conv2DLayer),
            ('pool', layers.MaxPool2DLayer),
            ('flatten', ReshapeLayer),  # output_dense
            ('encode_layer', layers.DenseLayer),
            ('hidden', layers.DenseLayer),  # output_dense
            ('unflatten', ReshapeLayer),
            ('unpool', Unpool2DLayer),
            ('deconv', layers.Conv2DLayer),
            ('output_layer', ReshapeLayer),
            ],
        input_shape=(None, 1, 80, 80),
        conv_num_filters=conv_filters,
        conv_filter_size=(filter_sizes, filter_sizes),
        conv_nonlinearity=None,
        pool_pool_size=(2, 2),
        flatten_shape=(([0], -1)), # not sure if necessary?
        encode_layer_num_units=encode_size,
        hidden_num_units=deconv_filters * (28 + filter_sizes - 1) ** 2 / 4,
        unflatten_shape=(([0], deconv_filters, (28 + filter_sizes - 1) / 2, (28 + filter_sizes - 1) / 2 )),
        unpool_ds=(2, 2),
        deconv_num_filters=1,
        deconv_filter_size=(filter_sizes, filter_sizes),
        # deconv_border_mode="valid",
        deconv_nonlinearity=None,
        output_layer_shape=(([0],-1)),
        update_learning_rate=0.01,
        update_momentum=0.975,
        batch_iterator_train=FlipBatchIterator(batch_size=128),
        regression=True,
        max_epochs=epochs,
        verbose=1,
        )
    ae.fit(xtrain, ytrain)

    X_train_pred = ae.predict(xtrain).reshape(-1, 80, 80)

if __name__ == "__main__":
    main()