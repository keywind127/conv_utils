from typing import *

from threading_utils import MultiThreadWorker

import numpy as np

import platform

import ctypes 

import os

class ConvTools(object):

    DEBUG_MODE = True

    LIBRARY_NAME_WINDOWS = "bin/conv_lib_windows.dll"

    LIBRARY_NAME_LINUX   = "bin/conv_lib_linux.so"

    LIBRARY_NAME = ((LIBRARY_NAME_WINDOWS) if (platform.system() == "Windows") else (LIBRARY_NAME_LINUX))

    LIBRARY_NAME = os.path.join(os.path.dirname(__file__), LIBRARY_NAME)

    LIBRARY = ctypes.cdll.LoadLibrary(LIBRARY_NAME)

    PARAM_TYPE = np.float32

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

    LIBRARY.transform_gradient.argtypes = [
        ctypes.c_void_p, # dstMatrix
        ctypes.c_void_p, # srcMatrix
        ctypes.c_void_p, # conMatrix
        ctypes.c_size_t, # batchSize (B)
        ctypes.c_size_t, # featureH (H)
        ctypes.c_size_t, # featureW (W)
        ctypes.c_size_t, # featureC (C)
        ctypes.c_size_t, # convC (CC)
        ctypes.c_size_t, # convW (CW)
        ctypes.c_size_t, # convH (CH)
        ctypes.c_size_t, # featureOutH (OW)
        ctypes.c_size_t, # featureOutW (OH)
        ctypes.c_size_t, # strideH (SH)
        ctypes.c_size_t  # strideW (SW)
    ]

    LIBRARY.transform_gradient_for_weights.argtypes = [
        ctypes.c_void_p, # srcMatrix
        ctypes.c_void_p, # dstMatrix
        ctypes.c_void_p, # conMatrix
        ctypes.c_size_t, # batchSize (B)
        ctypes.c_size_t, # featureH (H)
        ctypes.c_size_t, # featureW (W)
        ctypes.c_size_t, # featureC (C)
        ctypes.c_size_t, # convC (CC)
        ctypes.c_size_t, # convW (CW)
        ctypes.c_size_t, # convH (CH)
        ctypes.c_size_t, # featureOutH (OW)
        ctypes.c_size_t, # featureOutW (OH)
        ctypes.c_size_t, # strideH (SH)
        ctypes.c_size_t  # strideW (SW)
    ]

    LIBRARY.max_pooling_2D.argtypes = [
        ctypes.c_void_p, # dstMatrix
        ctypes.c_void_p, # binMatrix
        ctypes.c_void_p, # srcMatrix
        ctypes.c_size_t, # batchSize (B)
        ctypes.c_size_t, # featureC (C)
        ctypes.c_size_t, # featureH (H)
        ctypes.c_size_t, # featureW (W)
        ctypes.c_size_t, # featureOutH (OH),
        ctypes.c_size_t, # featureOutW (OW),
        ctypes.c_size_t, # poolH (PH)
        ctypes.c_size_t  # poolW (PW)
    ]

    @classmethod
    def max_pooling_2D(cls, src_matrix  : np.ndarray,
                            pool_size   : Tuple[ int, int ], *,
                            num_workers : Optional[ int ] = 4
                            
            ) -> Tuple[ np.ndarray, np.ndarray ]:
        
        if (cls.DEBUG_MODE):

            assert isinstance(src_matrix, np.ndarray)

            assert ((isinstance(pool_size, tuple)) or (isinstance(pool_size, list)))

            assert isinstance(num_workers, int)

            assert num_workers > 0
        
        if (src_matrix.dtype != cls.PARAM_TYPE):
            src_matrix = src_matrix.astype(cls.PARAM_TYPE)

        def process_data(src_matrix : np.ndarray,
                         pool_size  : Tuple[ int, int ]
                ) -> Tuple[ np.ndarray ]:

            (B, C, H, W) = src_matrix.shape

            (PH, PW) = pool_size

            (OH, OW) = (
                H // PH,
                W // PW
            )

            print(OH, OW)

            dst_matrix = np.zeros(shape = (B, C, OH, OW), dtype = cls.PARAM_TYPE)

            bin_matrix = np.zeros_like(src_matrix, dtype = cls.PARAM_TYPE)

            cls.LIBRARY.max_pooling_2D(
                dst_matrix.ctypes.data,
                bin_matrix.ctypes.data,
                src_matrix.ctypes.data,
                B,
                C,
                H,
                W,
                OH,
                OW,
                PH,
                PW
            )

            return (dst_matrix, bin_matrix)
        
        multi_thread_worker = MultiThreadWorker(num_workers, process_data)

        (dst_matrix, bin_matrix) = multi_thread_worker.process(src_matrix, pool_size, divide_axis = 0, concat_axis = 0)

        return (dst_matrix, bin_matrix)

    @classmethod
    def transform_gradient_for_weights(cls, src_matrix  : np.ndarray,
                                            dst_matrix  : np.ndarray,
                                            conv_matrix : np.ndarray,
                                            conv_stride : Tuple[ int, int ], *,
                                            num_workers : Optional[ int ] = 4) -> np.ndarray:
        
        if (cls.DEBUG_MODE):

            assert isinstance(src_matrix, np.ndarray)

            assert isinstance(dst_matrix, np.ndarray)

            assert isinstance(conv_matrix, np.ndarray)

            assert isinstance(conv_stride, tuple)

            assert isinstance(num_workers, int)

            assert num_workers > 0

        if (src_matrix.dtype != cls.PARAM_TYPE):
            src_matrix = cls.PARAM_TYPE(src_matrix)

        if (conv_matrix.dtype != cls.PARAM_TYPE):
            conv_matrix = cls.PARAM_TYPE(conv_matrix)

        if (dst_matrix.dtype != cls.PARAM_TYPE):
            dst_matrix = cls.PARAM_TYPE(dst_matrix)

        (*_, CH, CW) = conv_matrix.shape

        def process_data(dst_matrix  : np.ndarray, 
                         src_matrix  : np.ndarray, 
                         conv_stride : Tuple[ int, int ], 
                         CH          : int,
                         CW          : int) -> np.ndarray:

            (B, CC, OH, OW) = dst_matrix.shape

            (_, C, H, W) = src_matrix.shape

            (SH, SW) = conv_stride

            conv_gradients = np.zeros(shape = (CC, C, CH, CW), dtype = cls.PARAM_TYPE)

            cls.LIBRARY.transform_gradient_for_weights(
                src_matrix.ctypes.data,
                dst_matrix.ctypes.data,
                conv_gradients.ctypes.data,
                B,
                H,
                W,
                C,
                CC,
                CW,
                CH,
                OW,
                OH,
                SH,
                SW
            )

            return conv_gradients
        
        multi_thread_worker = MultiThreadWorker(num_workers, process_data)

        conv_gradients = multi_thread_worker.process(
            dst_matrix, 
            src_matrix, 
            conv_stride, 
            CH, 
            CW, 
            divide_axis = 1, 
            concat_axis = 0
        )

        return conv_gradients

    @classmethod
    def transform_gradient(cls, src_matrix  : np.ndarray,
                                conv_matrix : np.ndarray,
                                conv_stride : Tuple[ int, int ], *,
                                num_workers : Optional[ int ] = 4) -> np.ndarray:
        
        if (cls.DEBUG_MODE):

            assert isinstance(src_matrix, np.ndarray)

            assert isinstance(conv_matrix, np.ndarray)

            assert ((isinstance(conv_stride, tuple)) or (isinstance(conv_stride, list)))

            assert isinstance(num_workers, int)

            assert num_workers > 0

        if (src_matrix.dtype != cls.PARAM_TYPE):
            src_matrix = cls.PARAM_TYPE(src_matrix)

        if (conv_matrix.dtype != cls.PARAM_TYPE):
            conv_matrix = cls.PARAM_TYPE(conv_matrix)

        def process_data(src_matrix  : np.ndarray,
                         conv_matrix : np.ndarray,
                         conv_stride : Tuple[ int, int ]) -> np.ndarray:
            
            (SH, SW)        = conv_stride

            (CC, C, CH, CW) = conv_matrix.shape

            (B, _, OH, OW)  = src_matrix.shape

            (H, W) = (
                SH * (OH - 1) + CH,
                SW * (OW - 1) + CW
            )

            dst_matrix = np.zeros(shape = (B, C, H, W), dtype = cls.PARAM_TYPE)

            cls.LIBRARY.transform_gradient(
                dst_matrix.ctypes.data,
                src_matrix.ctypes.data,
                conv_matrix.ctypes.data,
                B,
                H,
                W,
                C,
                CC,
                CW,
                CH,
                OW,
                OH,
                SH,
                SW
            )

            return dst_matrix
        
        multi_thread_worker = MultiThreadWorker(num_workers, process_data)

        dst_matrix = multi_thread_worker.process(
            src_matrix, 
            conv_matrix, 
            conv_stride, 
            divide_axis = 0, 
            concat_axis = 0
        )

        return dst_matrix

    @classmethod
    def apply_convolution(cls, src_matrix  : np.ndarray,
                               conv_matrix : np.ndarray,
                               conv_stride : Tuple[ int, int ], *,
                               num_workers : Optional[ int ] = 4) -> np.ndarray:
        
        if (cls.DEBUG_MODE):

            assert isinstance(src_matrix, np.ndarray)

            assert isinstance(conv_matrix, np.ndarray)

            assert ((isinstance(conv_stride, tuple)) or (isinstance(conv_stride, list)))

        if (src_matrix.dtype != cls.PARAM_TYPE):
            src_matrix = cls.PARAM_TYPE(src_matrix)

        if (conv_matrix.dtype != cls.PARAM_TYPE):
            conv_matrix = cls.PARAM_TYPE(conv_matrix)
        
        def process_data(src_matrix  : np.ndarray,
                         conv_matrix : np.ndarray,
                         conv_stride : Tuple[ int, int ]) -> np.ndarray:
            
            (B, C, H, W) = src_matrix.shape

            (SH, SW) = conv_stride

            (CC, _, CH, CW) = conv_matrix.shape

            dst_matrix = np.zeros(shape = (B, CC, (H - CH) // SH + 1, (W - CW) // SW + 1), dtype = cls.PARAM_TYPE)

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

            return dst_matrix

        multi_thread_worker = MultiThreadWorker(num_workers, process_data)

        dst_matrix = multi_thread_worker.process(src_matrix, conv_matrix, conv_stride, divide_axis = 0, concat_axis = 0)

        return dst_matrix

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

        if (src_matrix.dtype != cls.PARAM_TYPE):
            src_matrix = cls.PARAM_TYPE(src_matrix)

        N = (1 + (H - CH) // SH) * (1 + (W - CW) // SW)

        dst_matrix = np.zeros(shape = (B, N, CH, CW, C), dtype = cls.PARAM_TYPE)

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

        return dst_matrix
    
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

    NUM_WORKERS = 8

    BATCH_SIZE = 32

    IMAGE_H = 24

    IMAGE_W = 24

    IMAGE_C = 128

    CONV_C_OUT = 256

    CONV_C_IN = IMAGE_C

    CONV_H = 3

    CONV_W = 3

    STRIDE_H = 1

    STRIDE_W = 1

    src_matrix = np.random.randint(-10, 10, (BATCH_SIZE, IMAGE_C, IMAGE_H, IMAGE_W))

    conv_matrix = np.random.randint(-10, 10, (CONV_C_OUT, CONV_C_IN, CONV_H, CONV_W))

    conv_stride = (STRIDE_H, STRIDE_W)

    from datetime import datetime

    SOT = datetime.now()

    dst_matrix = ConvTools.apply_convolution(src_matrix, conv_matrix, conv_stride, num_workers = NUM_WORKERS)

    print(f"Forward: {(datetime.now() - SOT).total_seconds()} seconds")

    (*_, OH, OW) = dst_matrix.shape

    from datetime import datetime

    SOT = datetime.now()

    gradient = ConvTools.transform_gradient(dst_matrix, conv_matrix, conv_stride, num_workers = NUM_WORKERS)

    print(f"Backward 1 (X): {(datetime.now() - SOT).total_seconds()} seconds, Shape: {gradient.shape}")

    SOT = datetime.now()

    gradient = ConvTools.transform_gradient_for_weights(src_matrix, dst_matrix, conv_matrix, conv_stride, num_workers = NUM_WORKERS)

    print(f"Backward 2 (W): {(datetime.now() - SOT).total_seconds()} seconds, Shape: {gradient.shape}")

    array = np.array([
        [ 
            [
                [ 1, 3, 2, 5, 1 ],
                [ 3, 5, 1, 3, 2 ],
                [ 5, 6, 3, 2, 4 ],
                [ 8, 5, 6, 3, 4 ]
            ]
        ]
    ])

    print(array.shape)

    dst_matrix, bin_matrix = ConvTools.max_pooling_2D(array, (2, 2), num_workers = 4)

    print(dst_matrix)

    print(bin_matrix)