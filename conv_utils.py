from typing import *

import numpy as np

import ctypes 

import os

class ConvTools(object):

    DEBUG_MODE = True

    LIBRARY_NAME = os.path.join(os.path.dirname(__file__), "conv.dll")

    LIBRARY = ctypes.cdll.LoadLibrary(LIBRARY_NAME)

    LIBRARY.extract_matrices.argtypes = [
        ctypes.c_void_p, # dstMatrix
        ctypes.c_void_p, # srcMatrix
        ctypes.c_size_t, # featureH (H)
        ctypes.c_size_t, # featureW (W)
        ctypes.c_size_t, # featureC (C)
        ctypes.c_size_t, # partialN (N)
        ctypes.c_size_t, # batchSize (B)
        ctypes.c_size_t, # convH (CH)
        ctypes.c_size_t, # convW (CW)
        ctypes.c_size_t, # strideH (SH)
        ctypes.c_size_t  # strideW (SW)
    ]

    LIBRARY.apply_convolution.argtypes = [
        ctypes.c_void_p, # dstMatrix
        ctypes.c_void_p, # srcMatrix
        ctypes.c_void_p, # convVector
        ctypes.c_size_t, # featureH (H)
        ctypes.c_size_t, # featureW (W)
        ctypes.c_size_t, # featureC (C)
        ctypes.c_size_t, # batchSize (B)
        ctypes.c_size_t, # convH (CH)
        ctypes.c_size_t, # convW (CW)
        ctypes.c_size_t, # convC (CC)
        ctypes.c_size_t, # strideH (SH)
        ctypes.c_size_t  # strideW (SW)
    ]

    @classmethod
    def apply_convolution(cls, src_matrix  : np.ndarray,
                               conv_matrix : np.ndarray,
                               conv_stride : Tuple[ int, int ]) -> np.ndarray:
        
        if (cls.DEBUG_MODE):

            assert isinstance(src_matrix, np.ndarray)

            assert isinstance(conv_matrix, np.ndarray)

            assert ((isinstance(conv_stride, tuple)) or (isinstance(conv_stride, list)))

        original_type = src_matrix.dtype

        if (src_matrix.dtype != np.float64):
            src_matrix = np.float64(src_matrix)

        if (conv_matrix.dtype != np.float64):
            conv_matrix = np.float64(conv_matrix)
        
        (B, C, H, W) = src_matrix.shape

        (SH, SW) = conv_stride

        (CC, _, CH, CW) = conv_matrix.shape

        dst_matrix = np.zeros(shape = (B, CC, (H - CH) // SH + 1, (W - CW) // SW + 1), dtype = np.float64)

        cls.LIBRARY.apply_convolution(
            dst_matrix.ctypes.data,
            src_matrix.ctypes.data,
            conv_matrix.ctypes.data,
            H,
            W,
            C,
            B,
            CH,
            CW,
            CC,
            SH,
            SW
        )

        return dst_matrix.astype(original_type)

    @classmethod
    def extract_matrices(cls, src_matrix  : np.ndarray, 
                              kernel_size : Tuple[ int, int ],
                              conv_stride : Tuple[ int, int ]
                              ) -> np.ndarray:
        
        if (cls.DEBUG_MODE):

            assert isinstance(src_matrix, np.ndarray)

            assert isinstance(kernel_size, tuple)

            assert isinstance(conv_stride, tuple)

            assert np.prod(src_matrix.shape) > 0

        (B, H, W, C) = src_matrix.shape

        (CH, CW) = kernel_size

        (SH, SW) = conv_stride

        original_type = src_matrix.dtype

        if (src_matrix.dtype != np.float64):
            src_matrix = np.float64(src_matrix)

        N = (1 + (H - CH) // SH) * (1 + (W - CW) // SW)

        dst_matrix = np.zeros(shape = (B, N, CH, CW, C), dtype = np.float64)

        cls.LIBRARY.extract_matrices(
            dst_matrix.ctypes.data,
            src_matrix.ctypes.data,
            H,  # height size
            W,  # width size
            C,  # channel size
            N,  # sub matrix quantity
            B,  # batch size 
            CH, # row kernel size 
            CW, # col kernel size
            SH, # row stride
            SW  # col stride
        )

        return dst_matrix.astype(original_type)
    
def extract_matrices(src_matrix : np.ndarray, kernel_size : tuple, conv_stride : tuple) -> np.ndarray:

    (B, H, W, C) = src_matrix.shape

    (CH, CW) = kernel_size

    (SH, SW) = conv_stride

    original_type = src_matrix.dtype

    if (src_matrix.dtype != np.float64):
        src_matrix = np.float64(src_matrix)

    N = (1 + (H - CH) // SH) * (1 + (W - CW) // SW)

    dst_matrix = np.zeros(shape = (B, N, CH, CW, C), dtype = np.float64)

    n_counter = 0

    for si in range(0, H - CH + 1, SH):

        for sj in range(0, W - CW + 1, SW):

            ei = si + CH

            ej = sj + CW

            # (B, N, CH, CW, C)
            dst_matrix[:,n_counter] = src_matrix[:,si:ei,sj:ej]

            n_counter += 1

    return dst_matrix.astype(original_type)

if (__name__ == "__main__"):

    # 10x10 => 0.0000000 sec
    # 50x50 => 0.0000000 sec
    # 100x100 => 0.00000 sec
    # 500x500 => 0.00518 sec
    # 1000x1000 => 0.018 sec

    BATCH_SIZE = 1

    IMAGE_H = 100

    IMAGE_W = 100

    IMAGE_C = 1

    CONV_C_OUT = 1

    CONV_C_IN = IMAGE_C

    CONV_H = 2

    CONV_W = 2

    STRIDE_H = 1

    STRIDE_W = 1

    src_matrix = np.random.randint(-10, 10, (BATCH_SIZE, IMAGE_C, IMAGE_H, IMAGE_W))

    conv_matrix = np.random.randint(-10, 10, (CONV_C_OUT, CONV_C_IN, CONV_H, CONV_W))

    conv_stride = (STRIDE_H, STRIDE_W)

    from datetime import datetime

    SOT = datetime.now()

    dst_matrix = ConvTools.apply_convolution(src_matrix, conv_matrix, conv_stride)

    print(f"{(datetime.now() - SOT).total_seconds()} seconds")

    print(dst_matrix.shape)