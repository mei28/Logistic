#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import theano
import theano.tensor as T


# In[ ]:


def main():
    breast_cancer = '/content/drive/My Drive/ロジスティック回帰/duke-breast-cancer.txt'
    data = pd.read_table(breast_cancer,header=None)
    X = data.drop(data.columns[0],axis=1)
    y = data[data.columns[0]]

    X_train, X_test, y_train, y_test=train_test_split(X, y, shuffle=True)
    
    lambd= 0.01
    training_epochs = 10

    w_init = np.random.normal(loc=0.0,scale=lambd,size=X_train.shape[1])
    train = model(X_train, y_train, lambd, w_init)

    for t in range(training_epochs):
        loss, w = train(X,y)
        print('{}: loss:{}'.format(t,loss))


# In[ ]:


class Optimizer(object):
  def __init__(self,params=None):
    if params is None:
      return NotImplementedError()
    self.params = params
  def updates(self,loss=None):
    if loss is None:
      return NotImplementedError()
    self.updates = OrderedDict()
    self.gparams = [T.grad(loss,param) for param in self.params]


# In[ ]:


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, params=None):
        super(SGD, self).__init__(params=params)
        self.learning_rate = 0.01

    def updates(self, loss=None):
        super(SGD, self).updates(loss=loss)

        for param, gparam in zip(self.params, self.gparams):
            self.updates[param] = param - self.learning_rate * gparam

        return self.updates


# In[ ]:


def model(X,y,lambd, w_init):
  X = T.matrix(name="X")
  y = T.vector(name="y")  
  w = theano.shared(w_init, name="w")
  
  p_1 = 1/(1+T.exp(-T.dot(X,w)))
  xent = -y * T.log(p_1) - (1-y)*T.log(1-p_1)
  loss = - xent.mean() + lambd * (w ** 2).sum()/2

  params = [w]
  updates = SGD(params=params).updates(loss)

  print('start: compile model')

  train = theano.function(
            inputs=[X, y],
            outputs=[loss,w],
            updates=updates)

  print('complete: compile model')

  return train


# In[ ]:


main()

