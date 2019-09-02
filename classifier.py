'''
Created on 15 Nov 2017

@author: ahmed.salem
Based on https://github.com/csong27/membership-inference/blob/master/classifier.py

'''

import sys
from sklearn.metrics import classification_report, accuracy_score
import theano.tensor as T
import numpy as np
import lasagne
import theano
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


sys.dont_write_bytecode = True


def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]




def get_cnn_model(n_in, n_hidden, n_out):
    net = dict()
    net['input'] = lasagne.layers.InputLayer(shape=(None, n_in[1], n_in[2], n_in[3]))
    
    net['conv1'] = lasagne.layers.Conv2DLayer(net['input'], num_filters=32, filter_size=(5, 5),
            pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    
    net['maxPool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], pool_size=(2, 2))
     
    net['conv2'] = lasagne.layers.Conv2DLayer(
            net['maxPool1'], num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    net['maxPool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], pool_size=(2, 2))

    net['fc'] = lasagne.layers.DenseLayer(
        net['maxPool2'],
        num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.tanh)
    
    net['output'] = lasagne.layers.DenseLayer(
        net['fc'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net



def get_nn_model(n_in, n_hidden, n_out):
    net = dict()
    print(n_in)
    net['input'] = lasagne.layers.InputLayer((None, n_in[1]))
    net['fc'] = lasagne.layers.DenseLayer(
        net['input'],
        num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.tanh)
    net['output'] = lasagne.layers.DenseLayer(
        net['fc'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net

def get_softmax_model(n_in, n_out):
    net = dict()
    net['input'] = lasagne.layers.InputLayer((None, n_in[1]))
    
    net['output'] = lasagne.layers.DenseLayer(
        net['input'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net


def train_model(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, model='cnn', l2_ratio=1e-7):

    
    train_x, train_y, test_x, test_y = dataset
    n_in = train_x.shape
    
    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)
    print('Building model with {} training data, {} classes...'.format(len(train_x), n_out))
    if model=='cnn' or model=='cnn2' or model=='Droppcnn' or  model=='Droppcnn2':
        input_var = T.tensor4('x')
    else:
        input_var = T.matrix('x')
    target_var = T.ivector('y')
    if model == 'cnn':
        print('Using a multilayer convolution neural network based model...')
        net = get_cnn_model(n_in, n_hidden, n_out)        
    elif model == 'nn':
        print('Using a multilayer neural network based model...')
        net = get_nn_model(n_in, n_hidden, n_out)
    else:
        print('Using a single layer softmax based model...')
        net = get_softmax_model(n_in, n_out)

    net['input'].input_var = input_var
    
    output_layer = net['output']
    # create loss function
    prediction = lasagne.layers.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + l2_ratio * lasagne.regularization.regularize_network_params(output_layer,
                                                                                 lasagne.regularization.l2)
    # create parameter update expressions
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # use trained network for predictions
    test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    test_fn = theano.function([input_var], test_prediction)
    print('Training...')
    counter = 1
    for epoch in range(epochs):
        loss = 0
        for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size):
            loss += train_fn(input_batch, target_batch)

        loss = round(loss, 3)
        if(epoch % 10 ==0):
            print('Epoch {}, train loss {}'.format(epoch, loss))
        
        
        counter = counter +1
    pred_y = []
    for input_batch, _ in iterate_minibatches(train_x, train_y, batch_size, shuffle=False):
        #input_batch = (np.reshape(input_batch,(len(input_batch),3,32,32)))
        pred = test_fn(input_batch)
        pred_y.append(np.argmax(pred, axis=1))
    pred_y = np.concatenate(pred_y)

    if test_x is not None:
        print('Testing...')
        pred_y = []

        if batch_size > len(test_y):
            batch_size = len(test_y)

        for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
            #input_batch = (np.reshape(input_batch,(len(input_batch),3,32,32)))
            pred = test_fn(input_batch)
            pred_y.append(np.argmax(pred, axis=1))
        pred_y = np.concatenate(pred_y)
        print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    

    print('More detailed results:')
    print(classification_report(test_y, pred_y))

        
    return output_layer
