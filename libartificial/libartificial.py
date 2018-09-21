import ctypes
import os

lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared', 'libartificial.so'))
