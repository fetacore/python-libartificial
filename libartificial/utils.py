import ctypes
import numpy as np
from .libartificial import lib

lib.init_wb.restype = ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
lib.init_wb.argtypes = [
	ctypes.c_double, #variance
	ctypes.c_int, #layers
	ctypes.POINTER(ctypes.c_int), #nodes[layers]
	ctypes.POINTER(ctypes.c_wchar_p), #functions[layers + 1],
	ctypes.c_int, #columns_Y
	ctypes.c_int #columns_X
]

lib.save_wb.restype = None
lib.save_wb.argtypes = [
  ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), #wb
  ctypes.c_int, #layers
  ctypes.POINTER(ctypes.c_int), #nodes[layers]
  ctypes.c_int, #columns_Y
  ctypes.c_int #columns_X
]

lib.load_wb.restype = ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
lib.load_wb.argtypes = [
  ctypes.c_int, #layers
  ctypes.POINTER(ctypes.c_int), #nodes[layers]
  ctypes.c_int, #columns_Y
  ctypes.c_int #columns_X
]

lib.delete_wb.restype = None
lib.delete_wb.argtypes = [
	ctypes.c_int, #layers
	ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), #wb
]

lib.delete_Z.restype = None
lib.delete_Z.argtypes = [
	ctypes.c_int, #layers
	ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), #Z
]

def init_wb(variance, nodes, funcs, columns_Y, columns_X):
  funcs = (ctypes.c_wchar_p * len(funcs))(*funcs)
  layers = len(nodes)
  nodes = (ctypes.c_int * len(nodes))(*nodes)
  wb = lib.init_wb(variance, layers, nodes, funcs, columns_Y, columns_X)
  return wb

def load_wb(nodes, columns_Y, columns_X):
  nodes = (ctypes.c_int * len(nodes))(*nodes)
  wb = lib.load_wb(len(nodes), nodes, columns_Y, columns_X)
  return wb

def save_wb(wb, nodes, columns_Y, columns_X):
  nodes = (ctypes.c_int * len(nodes))(*nodes)
  lib.save_wb(wb, len(nodes), nodes, columns_Y, columns_X)

def normalize(X):
  for i in range(0, len(X[0,:])):
    X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
  return(X)

def randomize(X):
  for i in range(0, len(X[0,:])):
    np.random.shuffle(X[:,i])
  return(X)

def freedom(Z, wb, layers):
  lib.delete_Z(layers, Z)
  lib.delete_wb(layers, wb)
