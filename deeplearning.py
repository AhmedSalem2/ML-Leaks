'''
Created on 15 Nov 2017

@author: ahmed.salem
Based on https://github.com/csong27/membership-inference/blob/master/attack.py
'''

import sys

sys.dont_write_bytecode = True


from classifier import train_model, iterate_minibatches
import numpy as np
import theano.tensor as T
import lasagne
import theano



np.random.seed(21312)



def train_target_model(dataset,epochs=100, batch_size=100, learning_rate=0.01, l2_ratio=1e-7,
                       n_hidden=50, model='nn'):

    
    train_x, train_y, test_x, test_y = dataset

    output_layer = train_model(dataset, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                               batch_size=batch_size, model=model, l2_ratio=l2_ratio)
    # test data for attack model
    attack_x, attack_y = [], []
    if model=='cnn':
        #Dimension for CIFAR-10
        input_var = T.tensor4('x')
    else:
        #Dimension for News
        input_var = T.matrix('x')

    prob = lasagne.layers.get_output(output_layer, input_var, deterministic=True)

    prob_fn = theano.function([input_var], prob)
    
    # data used in training, label is 1
    for batch in iterate_minibatches(train_x, train_y, batch_size, False):
        attack_x.append(prob_fn(batch[0]))
        attack_y.append(np.ones(len(batch[0])))
        
    # data not used in training, label is 0
    for batch in iterate_minibatches(test_x, test_y, batch_size, False):
        attack_x.append(prob_fn(batch[0]))
        attack_y.append(np.zeros(len(batch[0])))
        
    #print len(attack_y)
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    #print('total length  ' + str(sum(attack_y)))
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    return attack_x, attack_y, output_layer


