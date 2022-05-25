import numpy
import ctypes

PyLong_AsByteArray = ctypes.pythonapi._PyLong_AsByteArray
PyLong_AsByteArray.argtypes = [ctypes.py_object,
                               numpy.ctypeslib.ndpointer(numpy.uint8),
                               ctypes.c_size_t,
                               ctypes.c_int,
                               ctypes.c_int]


def int_to_bytes(num: int):
    a = numpy.zeros(num.bit_length() // 8 + 1, dtype=numpy.uint8)
    PyLong_AsByteArray(num, a, a.size, 0, 1)
    return a
