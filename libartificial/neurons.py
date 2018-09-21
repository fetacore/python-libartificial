import ctypes
import numpy as np
from .libartificial import lib

lib.cpu_feedforward.restype = ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
lib.cpu_feedforward.argtypes = [
  ctypes.c_int, #rows
	ctypes.c_int, #columns_Y
	ctypes.c_int,	#columns_X
	ctypes.c_int, #layers
  ctypes.POINTER(ctypes.c_double), #X
  ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), #wb
	ctypes.POINTER(ctypes.c_int), #nodes[layers]
	ctypes.POINTER(ctypes.c_wchar_p) #functions[layers + 1]
]

def ff(Y, X, wb, nodes, funcs):
  funcs = (ctypes.c_wchar_p * len(funcs))(*funcs)
  [rows, columns_Y] = Y.shape
  [rows, columns_X] = X.shape
  layers = len(nodes)
  nodes = (ctypes.c_int * len(nodes))(*nodes)
  X = X.astype(np.double)
  X = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  feed = lib.cpu_feedforward(rows, columns_Y, columns_X, layers, X, wb, nodes, funcs)
  return feed
