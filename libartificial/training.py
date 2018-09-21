import ctypes
import numpy as np
from .libartificial import lib

#lib.cpu_feedforward_update.restype = None
#lib.cpu_feedforward_update.argtypes = [
	#ctypes.c_int, #rows
	#ctypes.c_int, #columns_Y
	#ctypes.c_int,	#columns_X
	#ctypes.c_int, #layers
	#ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), #Z
  #ctypes.POINTER(ctypes.c_double), #X
  #ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), #wb
	#ctypes.POINTER(ctypes.c_int), #nodes[layers]
	#ctypes.POINTER(ctypes.c_wchar_p) #functions[layers + 1]
#]

lib.cpu_gd_update.restype = None
lib.cpu_gd_update.argtypes = [
	ctypes.c_int, #rows
	ctypes.c_int, #columns_Y
	ctypes.c_int,	#columns_X
	ctypes.c_int, #batch
	ctypes.c_int, #layers
	ctypes.POINTER(ctypes.c_int), #nodes[layers]
	ctypes.POINTER(ctypes.c_double), #Y
	ctypes.POINTER(ctypes.c_double), #X
	ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), #Z
  ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), #wb
	ctypes.POINTER(ctypes.c_wchar_p), #functions[layers + 1]
	ctypes.c_double, #learning rate
	ctypes.c_int #epochs
]

#lib.cpu_gd_delta.restype = None
#lib.cpu_gd_delta.argtypes = [
  #ctypes.POINTER(ctypes.c_double), #deltas
	#ctypes.c_int, #rows
	#ctypes.c_int, #columns_Y
	#ctypes.c_int, #layers
	#ctypes.POINTER(ctypes.c_double), #Y_row
	#ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), #Z_row
  #ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), #wb
  #ctypes.POINTER(ctypes.c_int), #nodes[layers]
	#ctypes.POINTER(ctypes.c_wchar_p) #functions[layers + 1]
#]

def update(Y, X, Z, wb, nodes, funcs, batch, learning_rate, epochs):
  funcs = (ctypes.c_wchar_p * len(funcs))(*funcs)
  [rows, columns_Y] = Y.shape
  [rows, columns_X] = X.shape
  layers = len(nodes)
  nodes = (ctypes.c_int * len(nodes))(*nodes)
  Y = Y.astype(np.double)
  Y = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  X = X.astype(np.double)
  X = X.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
  lib.cpu_gd_update(rows, columns_Y, columns_X, batch, layers, nodes, Y, X, Z, wb, funcs, learning_rate, epochs)
  return None
