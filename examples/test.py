import libartificial as ai
import numpy as np
import random

def comb_arrays(x):
  #x is the list of arrays (for multivariate environment)
  z = np.zeros((len(x[0][:]),len(x)))
  for i in range(0,len(x)):
    for j in range(0,len(x[0])):
      z[j,i] = x[i][j]
  return(z)

n = 1024
columns_X = 1
Xlist = []
Ylist = []
true_model = np.zeros((n,1))
for p in range(0, columns_X):
  x = np.random.normal((random.randint(-100, 100)),(random.randint(1, 20)),size=(n,1))
  Xlist.append(x)
  true_model = true_model + x**2
error = np.random.normal(0,1,size=(n,1))
Y = true_model + error
Ylist.append(Y)
Y = comb_arrays(Ylist)
X = comb_arrays(Xlist)

variance = 0.01
epochs = 3
batch = 256
eta = 0.000001
hlayers = [1056, 552, 5450, 360, 230, 5405, 340, 593]
active_fnct = [
  'logistic',
  'tanh',
  'gauss',
  'softsign',
  'softmax',
  'softplus',
  'tanh',
  'logistic',
  'linear'
]

X = ai.utils.normalize(X)
X = ai.utils.randomize(X)
wb = ai.utils.init_wb(variance, hlayers, active_fnct, 1, columns_X)
#wb = ai.utils.load_wb(hlayers, 1, columns_X)

feed = ai.neurons.ff(Y, X, wb, hlayers, active_fnct)
ai.training.update(Y, X, feed, wb, hlayers, active_fnct, batch, eta, epochs)

#ai.utils.save_wb(wb, hlayers, 1, columns_X)
ai.utils.freedom(feed, wb, len(hlayers))
