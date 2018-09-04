from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange


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
    
        #Cross_entrophy
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
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }
    
    #Use Simulated anneling to train this model (Metropolis method)
    def train_sa(self, X, y, X_val, y_val,
            reg=5e-6, num_iters=100, step_len = 0.01,
            batch_size=200, T_max=10, T_min=0.1, verbose=False):
        
      
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
        
            self.params['W1'] = W1 + step_len * np.random.uniform(-1, 1, W1.shape) * T
            self.params['b1'] = b1 + step_len * np.random.uniform(-1, 1, b1.shape) * T
            self.params['W2'] = W2 + step_len * np.random.uniform(-1, 1, W2.shape) * T
            self.params['b2'] = b2 + step_len * np.random.uniform(-1, 1, b2.shape) * T
            
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
                
        return {
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
            ratio += (dist2 - dist1) / (2 * sigma * sigma * step_len * step_len)
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