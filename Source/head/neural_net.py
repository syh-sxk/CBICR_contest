from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

#import sys
#sys.path.append('E:\\Documents\\1资料\\Brainmatrix\\模拟退火\\CBICR_contest\\Source\\head\\utils\\')
from head.utils.layers import *
from head.utils.fast_layers import *
from head.utils.layer_utils import *


class TwoLayerNet(object):
    
    __INF = 1e10 

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
  
    #forward_pass
    def __fwd_pass(self, X, with_hidden = False):
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
        if with_hidden:
            return np.dot(hidden_layer, W2) + b2 , hidden_layer
        else:
            return np.dot(hidden_layer, W2) + b2
    
    def loss(self, X, y=None, reg=0.0, with_grads = False):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        scores, hidden_layer = self.__fwd_pass(X, with_hidden = True)
    
        #Cross_entropy
        loss = None
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
        correct_logprobs = -np.log(probs[np.arange(N),y])
        data_loss = np.sum(correct_logprobs)/N
    
        #Regression
        reg_loss = reg*np.sum(W1*W1)+reg*np.sum(W2*W2)
        loss = data_loss+reg_loss
        
        if not with_grads :
            return loss
         
        #BP
        grads = {}
        dscores = probs
        dscores[np.arange(N),y] -= 1
        dscores /= N

        grads['W2'] = np.dot(hidden_layer.T,dscores)
        grads['b2'] = np.sum(dscores,axis=0,keepdims=True)

        # Backpropagate the gradient to the hidden layer
        dhidden = np.dot(dscores,W2.T)

        # Kill the gradient flow where ReLU activation was clamped at 0
        dhidden[hidden_layer==0] = 0

        # Backpropagate the gradient to W and b
        grads['W1'] = np.dot(X.T,dhidden)
        grads['b1'] = np.sum(dhidden, axis=0, keepdims=True)

        # Add the gradient of the regularization
        grads['W1'] += 2 * reg * W1
        grads['W2'] += 2 * reg * W2

        return loss, grads
        
    def predict(self, X):
        y_pred = self.__fwd_pass(X)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
    
    def __mini_batch(self, X, y, num_train, batch_size):
        idx_batch = np.random.choice(np.arange(num_train),size = batch_size)
        X_batch = X[idx_batch]
        y_batch = y[idx_batch]
        return X_batch, y_batch
        
    def __normsqr(self, X):
        return np.dot(X.ravel(), X.ravel())
        
    # Use SGD to optimize the parameters in self.model    
    def train_bp(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
        
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):
    
            X_batch, y_batch = self.__mini_batch(X, y, num_train, batch_size)
        
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg, with_grads = True)
            loss_history.append(loss)
        
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b1'] -= learning_rate * grads['b1'].ravel()
            self.params['b2'] -= learning_rate * grads['b2'].ravel()

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
                print('iteration %d / %d: train_acc %f' % ((self.predict(X_batch) == y_batch).mean()))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
        
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'params': self.params,
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }
    
    #Use Simulated anneling to train this model (Metropolis method)
    def train_sa(self, X, y, X_val, y_val,
            reg=5e-6, num_iters=100, step_len = 0.01,
            batch_size=200, T_max=1, T_min=0.001, verbose=False):
        
        acc = 0
        rej = 0
        T = np.copy(T_max)
        Tfactor = -np.log(T_max / T_min)
        
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        loss_past = loss_new = self.__INF
        
        for it in xrange(num_iters):
    
            W1, b1 = self.params['W1'], self.params['b1']
            W2, b2 = self.params['W2'], self.params['b2']
  
            X_batch, y_batch = self.__mini_batch(X, y, num_train, batch_size)
            
            # multiply T
            self.params['W1'] = W1 + step_len * np.random.uniform(-1, 1, W1.shape) * T
            self.params['b1'] = b1 + step_len * np.random.uniform(-1, 1, b1.shape) * T
            self.params['W2'] = W2 + step_len * np.random.uniform(-1, 1, W2.shape) * T
            self.params['b2'] = b2 + step_len * np.random.uniform(-1, 1, b2.shape) * T
            
            loss_new = self.loss(X_batch, y = y_batch, reg = reg)
            
            #print('it:', it)
            #print('loss_past:', loss_past, '  loss_new:', loss_new)
            #Metropolis Method
            ratio = np.exp((loss_past - loss_new) / T)
            #print('T:', T, 'ratio:', ratio) 
            thres = np.random.uniform(0,1)
            if ratio < thres : # Reject new solution
                self.params['W1'] = W1
                self.params['b1'] = b1
                self.params['W2'] = W2
                self.params['b2'] = b2
                rej += 1
            else : #Accept new solution
                loss_past = loss_new
                acc += 1

            #stats loging
            loss_history.append(loss_past)
            
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss_past))
                print('iteration %d / %d: train_acc %f' % ((self.predict(X_batch) == y_batch).mean()))
                print('Reject:', rej, '  Accept:', acc)
                
                #print(train_acc)

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
        
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                
            T = T_max * np.exp(Tfactor * it / num_iters)
                
        return {
           'params': self.params,
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }
    
    #Experiment by SYH
    def train_test(self, X, y, X_val, y_val,
            reg=0.01, num_iters=100, step_len = 5e-4, sigma = 1e-10,
            batch_size=200, T_max=1, T_min=1, verbose=False):
        
        T = np.copy(T_max)
        Tfactor = -np.log(T_max / T_min)
        
        acc = rej = 0 # Counting accept/reject rate
        
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        for it in xrange(num_iters):
            
            W1, b1 = self.params['W1'], self.params['b1']
            W2, b2 = self.params['W2'], self.params['b2']
        
            
            X_batch, y_batch = self.__mini_batch(X, y, num_train, batch_size)
        
            loss_past , grads_past = self.loss(X_batch, y = y_batch, reg = reg, with_grads = True)
            
            W1_rand = np.random.normal(0, step_len * sigma, W1.shape)
            b1_rand = np.random.normal(0, step_len * sigma, b1.shape)
            W2_rand = np.random.normal(0, step_len * sigma, W2.shape)
            b2_rand = np.random.normal(0, step_len * sigma, b2.shape)
            
            self.params['W1'] = W1 + W1_rand - step_len * grads_past['W1']
            self.params['b1'] = b1 + b1_rand - step_len * grads_past['b1'].ravel()
            self.params['W2'] = W2 + W2_rand - step_len * grads_past['W2']
            self.params['b2'] = b2 + b2_rand - step_len * grads_past['b2'].ravel()
            
            loss_new, grads_new = self.loss(X_batch, y = y_batch, reg = reg, with_grads = True)
            
            W1_dist_new = self.params['W1'] - step_len * grads_new['W1'] - W1
            W2_dist_new = self.params['W2'] - step_len * grads_new['W2'] - W2
            b1_dist_new = self.params['b1'] - step_len * grads_new['b1'] - b1
            b2_dist_new = self.params['b2'] - step_len * grads_new['b2'] - b2
            
            #Metropolis Method
            dist1 = self.__normsqr(W1_rand) + self.__normsqr(b1_rand) + self.__normsqr(W2_rand) + self.__normsqr(b2_rand) #f(i -> j)
            dist2 = self.__normsqr(W1_dist_new) + self.__normsqr(W2_dist_new) + self.__normsqr(b1_dist_new) + self.__normsqr(b2_dist_new) #f(j -> i)
            ratio = (loss_past - loss_new) / T
            #ratio_tmp = ratio
            ratio += (dist2 - dist1) / (2 * sigma * sigma * step_len * step_len)
            #print('Before considering dist:', ratio_tmp, '  After considering dist:', ratio)
            
            if ratio < 0: #Ratio could be very big number, a precaution for exp
                ratio = np.exp(ratio)
            else :
                ratio = 1
            
            thres = np.random.uniform(0,1)
            if ratio < thres : # Reject new solution
                self.params['W1'] = W1
                self.params['b1'] = b1
                self.params['W2'] = W2
                self.params['b2'] = b2
                rej += 1
            else : #Accept new solution
                loss_past = loss_new
                acc += 1
            
            #stats loging
            loss_history.append(loss_past)
            
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss_past))
                #print(train_acc)

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
        
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                
            T = T_max * np.exp(Tfactor * it / num_iters)
        
        print('Accept: %d   Reject: %d' % (acc, rej))
        
        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }
    
    
    # Experiment by SXK
    def train_bp_sa(self, X, y, X_val, y_val, 
            learning_rate_max = 1e-3, learning_rate_min = 1e-4, reg=0.1, 
            num_iters_per_sgd=1500, num_sgds = 2, num_iters_per_sa = 500, 
            step_len = 0.001, T_max=0.1, T_min=0.005, if_sa = True,
            batch_size=200, verbose=False):
     
        Tfactor = -np.log(T_max / T_min)
        
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        
        for it_sgd in xrange(num_sgds-1):
            for it in xrange(num_iters_per_sgd):

                X_batch, y_batch = self.__mini_batch(X, y, num_train, batch_size)
                
                loss, grads = self.loss(X_batch, y=y_batch, reg=reg, with_grads = True)
                loss_history.append(loss)
                
                # SGD
                learning_rate = (learning_rate_min + learning_rate_max) / 2 + (
                                 learning_rate_max - learning_rate_min) / 2 * np.cos(
                                 it / num_iters_per_sgd * np.pi)
                
                self.params['W1'] -= learning_rate * grads['W1']
                self.params['W2'] -= learning_rate * grads['W2']
                self.params['b1'] -= learning_rate * grads['b1'].ravel()
                self.params['b2'] -= learning_rate * grads['b2'].ravel()
                
                if verbose and it % 100 == 0:
                    print('it_sgd sgd %d / %d, iteration %d / %d: loss %4.2f, train_acc %4.2f' % (
                            it_sgd+1, num_sgds, it, num_iters_per_sgd, loss,
                            (self.predict(X_batch) == y_batch).mean()))
                
                if it % iterations_per_epoch == 0:
                    # Check accuracy
                    train_acc = (self.predict(X_batch) == y_batch).mean()
                    val_acc = (self.predict(X_val) == y_val).mean()
                    train_acc_history.append(train_acc)
                    val_acc_history.append(val_acc)
                
            # SA
            if if_sa:
                loss_past = loss_new = loss
                T = np.copy(T_max)
                for it in xrange(num_iters_per_sa):
                    W1, b1 = self.params['W1'], self.params['b1']
                    W2, b2 = self.params['W2'], self.params['b2']
                    self.params['W1'] = W1 + step_len * np.random.uniform(-1, 1, W1.shape) * T
                    self.params['b1'] = b1 + step_len * np.random.uniform(-1, 1, b1.shape) * T
                    self.params['W2'] = W2 + step_len * np.random.uniform(-1, 1, W2.shape) * T
                    self.params['b2'] = b2 + step_len * np.random.uniform(-1, 1, b2.shape) * T


                    loss_new, grads_new = self.loss(X_batch, y = y_batch, reg = reg, with_grads = True)

                    loss_new = self.loss(X_batch, y = y_batch, reg = reg)

                    #Metropolis Method
                    ratio = np.exp((loss_past - loss_new) / T)
                    thres = np.random.uniform(0,1)
                    if ratio < thres : # Reject new solution
                        self.params['W1'] = W1
                        self.params['b1'] = b1
                        self.params['W2'] = W2
                        self.params['b2'] = b2
                    else : #Accept new solution
                        loss_past = loss_new

                    #stats loging
                    loss_history.append(loss_past)

                    if verbose and it % 100 == 0:
                        print('it_sgd sa %d / %d, iteration %d / %d: loss %4.2f, train_acc %4.2f' % (
                                it_sgd+1, num_sgds, it, num_iters_per_sa, loss_past,
                                (self.predict(X_batch) == y_batch).mean()))

                    # Every epoch, check train and val accuracy and decay learning rate.
                    if it % iterations_per_epoch == 0:
                        # Check accuracy
                        train_acc = (self.predict(X_batch) == y_batch).mean()
                        val_acc = (self.predict(X_val) == y_val).mean()
                        train_acc_history.append(train_acc)
                        val_acc_history.append(val_acc)

                    T = T_max * np.exp(Tfactor * it / num_iters_per_sa)
                
                
        for it in xrange(num_iters_per_sgd):
            X_batch, y_batch = self.__mini_batch(X, y, num_train, batch_size)

            loss, grads = self.loss(X_batch, y=y_batch, reg=reg, with_grads = True)
            loss_history.append(loss)

            # SGD
            learning_rate = (learning_rate_min + learning_rate_max) / 2 + (
                             learning_rate_max - learning_rate_min) / 2 * np.cos(
                             it / num_iters_per_sgd * np.pi)

            self.params['W1'] -= learning_rate * grads['W1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b1'] -= learning_rate * grads['b1'].ravel()
            self.params['b2'] -= learning_rate * grads['b2'].ravel()

            if verbose and it % 100 == 0:
                print('it_sgd sgd %d / %d, iteration %d / %d: loss %4.2f, train_acc %4.2f' % (
                        num_sgds, num_sgds, it, num_iters_per_sgd, loss,
                        (self.predict(X_batch) == y_batch).mean()))

            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }
    

class FourLayerConvNet_fast(object):
    # Add batch normalization
    def __init__(self, input_dim=(3, 32, 32), num_filters=(32, 64, 128), filter_size=(7, 3, 3),
                 num_classes=10, weight_scale=1e-3, reg=0.0, use_batchnorm = False,
                 dtype=np.float32):
        
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Get input dimensions
        C, H, W = input_dim

        # Compute max pooling filter dimensions
        self.conv_param = {'stride': 1, 'pad': 1}
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        HP = (H-self.pool_param['pool_height'])/self.pool_param['stride']+1
        WP = (W-self.pool_param['pool_width'])/self.pool_param['stride']+1

        # Set weights and biases dimension
        weights_dim = [(num_filters[0], C, filter_size[0], filter_size[0]),
                        (num_filters[1], num_filters[0], filter_size[1], filter_size[1]),
                        (num_filters[2], num_filters[1], filter_size[2], filter_size[2]),
                        (num_filters[2], num_classes)]
        biases_dim = [num_filters[0], num_filters[1], num_filters[2], num_classes]

        num_params = num_filters[0]*C*filter_size[0]**2 + num_filters[1]*num_filters[0]*filter_size[1]**2 + \
                 num_filters[2]*num_filters[1]*filter_size[2]**2 + num_filters[2]*num_classes + \
                 num_filters[0] + num_filters[1] + num_filters[2] + num_classes
                
        print('Number of parameters: ', num_params)
        
        # Calculate the dimensions of each feature maps (to decide the dimensions of BN params).
        if use_batchnorm:
            #Assert input is square
            x_size1 = ((input_dim[1]+2*self.conv_param['pad']-filter_size[1])//self.conv_param['stride']+1)**2 
            x_dim1 = int(num_filters[0] * x_size1)

            #Assert pool size is square
            x_size2 = ((x_size1 - self.pool_param['pool_height']) // self.pool_param['stride'] + 1) ** 2 
            x_size2 = ((x_size2+2*self.conv_param['pad']-filter_size[2])//self.conv_param['stride']+1)**2 
            x_dim2 = int(num_filters[1] * x_size2)

            x_size3 = ((x_size2 - self.pool_param['pool_height']) // self.pool_param['stride'] + 1) ** 2
            x_size3 = ((x_size3+2*self.conv_param['pad']-filter_size[2])//self.conv_param['stride']+1)**2 
            x_dim3 = int(num_filters[2] * x_size3)

            x_dims = [x_dim1, x_dim2, x_dim3, num_filters[2]]

        # Initialize weights and biases
        num_layers = len(weights_dim)
        for i in range(1,num_layers+1):
            self.params['W%d' %i] = np.random.normal(loc=0.0, scale=weight_scale,
                                                    size=weights_dim[i-1])
            self.params['b%d' %i] = np.zeros(biases_dim[i-1])
            
            if use_batchnorm:
                self.params['gamma%d' %i] = np.ones(x_dims[i-1])
                self.params['beta%d' %i] = np.zeros(x_dims[i-1])

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
            
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm

        
    def loss(self, X, y=None):
        mode = 'test' if y is None else 'train'
        
        bn_param = {}
        bn_param['mode'] = mode
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        
        W_list = [W1, W2, W3, W4]
        b_list = [b1, b2, b3, b4]
        
        use_batchnorm = self.use_batchnorm
        if use_batchnorm:
            gamma1, beta1 = self.params['gamma1'], self.params['beta1']
            gamma2, beta2 = self.params['gamma2'], self.params['beta2']
            gamma3, beta3 = self.params['gamma3'], self.params['beta3']
            
            gamma_list = [gamma1, gamma2, gamma3]
            beta_list = [beta1, beta2, beta3]

        # pass conv_param to the forward pass for the convolutional layer
        conv_param = self.conv_param

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = self.pool_param

        scores = None
        
        conv_cache = {}
        conv_relu_cache = {}
        max_cache = {}
        batch_cache = {}
        x = X.copy()
        for i in range(self.num_layers-2):
            x, conv_cache[i] = conv_forward_fast(x, W_list[i], b_list[i], conv_param)
            
            if use_batchnorm:
                x, batch_cache[i] = batchnorm_forward(x, gamma_list[i], beta_list[i], bn_param)
                
            x, conv_relu_cache[i] = relu_forward(x)
            x, max_cache[i] = max_pool_forward_fast(x, pool_param)
            
        x, conv_cache[self.num_layers-2] = conv_forward_fast(
            x, W_list[self.num_layers-2], b_list[self.num_layers-2], conv_param)
        
        if use_batchnorm:
            x, batch_cache[self.num_layers-2] = batchnorm_forward(
                x, gamma_list[self.num_layers-2], beta_list[self.num_layers-2], bn_param)
        
        x, conv_relu_cache[self.num_layers-2] = relu_forward(x)

        _, _, pool_height_last, pool_width_last = x.shape
        pool_param_last = {'pool_height': pool_height_last, 'pool_width': pool_width_last, 'stride': 1}
        x, max_cache[self.num_layers-2] = max_pool_forward_fast(x, pool_param_last)

        # Compute scores
        scores, scores_cache = affine_forward(x, W_list[-1], b_list[-1])

        if y is None:
            return scores

        loss, grads = 0, {}

        # Compute loss and gradient with respect to the softmax function
        loss, dout = softmax_loss(scores, y)

        # Add L2 regularization to the loss function
        for i in range(self.num_layers):
            loss += 0.5 * self.reg * np.sum(W_list[i] * W_list[i])

        # Backward
        daffine, grads['W4'], grads['b4'] = affine_backward(dout, scores_cache)
        dmax_pool = max_pool_backward_fast(daffine, max_cache[2])  

        dX = relu_backward(dmax_pool, conv_relu_cache[2])
        
        if use_batchnorm:
            dX, grads['gamma3'], grads['beta3'] = batchnorm_backward(dX, batch_cache[2])
            
        dX, grads['W3'], grads['b3'] = conv_backward_fast(dX, conv_cache[2])
                       
        dmax_pool = max_pool_backward_fast(dX, max_cache[1])             
        dX = relu_backward(dmax_pool, conv_relu_cache[1])
        
        if use_batchnorm:
            dX, grads['gamma2'], grads['beta2'] = batchnorm_backward(dX, batch_cache[1])
            
        dX, grads['W2'], grads['b2'] = conv_backward_fast(dX, conv_cache[1])

        dmax_pool = max_pool_backward_fast(dX, max_cache[0])           
        dX = relu_backward(dmax_pool, conv_relu_cache[0])
        
        if use_batchnorm:
            dX, grads['gamma1'], grads['beta1'] = batchnorm_backward(dX, batch_cache[0])
            
        dX, grads['W1'], grads['b1'] = conv_backward_fast(dX, conv_cache[0])


        # Add regularization to the gradients
        grads['W4'] += self.reg*W4
        grads['W3'] += self.reg*W3
        grads['W2'] += self.reg*W2
        grads['W1'] += self.reg*W1

        return loss, grads