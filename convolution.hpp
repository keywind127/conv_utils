#ifndef H_CONVOLUTION

#include "parameters.hpp"

#include <cstdio>

namespace Convolution
{

    inline size_t get_src_index(size_t i, size_t k, size_t n, size_t featureC, size_t featureW, size_t featureH)
    {

        // i : batchIndex
        // k : featureChannelIndex
        // n : pixelIndex

        size_t featureHW = featureH * featureW;

        return n + k * featureHW + i * featureC * featureHW;

    }

    inline size_t get_dst_index(size_t i, size_t j, size_t l, size_t m, size_t convC, size_t dstW, size_t dstH, size_t strideH, size_t strideW)
    {

        // dstMatrix rowIndex
        size_t dL = l / strideH;

        // dstMatrix colIndex
        size_t dM = m / strideW;

        // product of width and height of dstMatrix
        size_t dstHW = dstH * dstW;

        return dM + dL * dstW + j * dstHW + i * convC * dstHW;

    }

    inline size_t get_con_index(size_t j, size_t k, size_t p, size_t q, size_t featureC, size_t convW, size_t convH)
    {

        // j : outputChannelIndex
        // k : inputChannelIndex
        // p : vectorWeightRowIndex
        // q : vectorWeightColIndex

        size_t convHW = convH * convW;

        return q + p * convW + k * convHW + j * featureC * convHW;

    }

    void apply_convolution(
                           PARAM_TYPE * dstMatrix,  // (batchSize, convC, (featureH - convH) / strideH + 1, (featureW - convW) / strideW + 1)
                           PARAM_TYPE * srcMatrix,  // (batchSize, featureC, featureH * featureW)
                           PARAM_TYPE * convVector, // (convC, featureC, convH, convW)
                           size_t featureH,
                           size_t featureW,
                           size_t featureC,
                           size_t batchSize,
                           size_t convH,
                           size_t convW,
                           size_t convC,
                           size_t strideH,
                           size_t strideW
                        ) {

        // height of output feature map
        size_t dstH = (featureH - convH) / strideH + 1;

        // width of output feature map
        size_t dstW = (featureW - convW) / strideW + 1;

        // indexes for iteration
        size_t i, j, k, l, m, n, o, p, q;

        // exclusive row border of input feature map
        size_t rowBorder = featureH - convH + 1;

        // exclusive col border of input feature map
        size_t colBorder = featureW - convW + 1;

        // size of convolution matrix weight vector (last axis)
        // size_t conVectorSize = featureW * (convH - 1) + convW;

        // indexes for three different matrices
        size_t srcIndex, dstIndex, conIndex;

        // iterate through batches
        for (i = 0; i < batchSize; ++i)
        {

            // iterate through output channels
            for (j = 0; j < convC; ++j)
            {

                // iterate through input channels
                for (k = 0; k < featureC; ++k)
                {

                    // select row starting index
                    for (l = 0; l < rowBorder; l += strideH)
                    {

                        // select col starting index
                        for (m = 0; m < colBorder; m += strideW)
                        {

                            // calculate 1D index equivalent of (L, M)
                            // n = l * featureW + m;

                            // calculate dstMatrix index
                            dstIndex = get_dst_index(i, j, l, m, convC, dstW, dstH, strideH, strideW);

                            // initialize dstMatrix index position with 0
                            // dstMatrix[dstIndex] = 0;

                            for (p = 0; p < convH; ++p)
                            {

                                for (q = 0; q < convW; ++q)
                                {

                                    n = (l + p) * featureW + (m + q);

                                    srcIndex = get_src_index(i, k, n, featureC, featureW, featureH);

                                    conIndex = get_con_index(j, k, p, q, featureC, convW, convH);

                                    dstMatrix[dstIndex] += (convVector[conIndex] * srcMatrix[srcIndex]);

                                }

                            }

                        }

                    }

                }

            }

        }

    }

    void transform_gradient(
                            PARAM_TYPE * dstMatrix, // (batchSize, featureC, featureH, featureW)
                            PARAM_TYPE * srcMatrix, // (batchSize, convC, featureOutH, featureOutW)
                            PARAM_TYPE * conMatrix, // (convC, featureC, convH, convW)
                            size_t batchSize,
                            size_t featureH,
                            size_t featureW,
                            size_t featureC,
                            size_t convC,
                            size_t convW,
                            size_t convH,
                            size_t featureOutH,
                            size_t featureOutW,
                            size_t strideH,
                            size_t strideW
                            ) {

        size_t _CI, _CO, _B, _H, _W, _I, _J, __I, __J;

        size_t dstH = (featureH - convH) / strideH + 1;

        size_t dstW = (featureW - convW) / strideW + 1;

        size_t borderI = (featureH - convH) + 1;

        size_t borderJ = (featureW - convW) + 1;

        size_t srcIndex, dstIndex, conIndex;

        for (_CI = 0; _CI < featureC; ++_CI)
        {

            for (_CO = 0; _CO < convC; ++_CO)
            {

                for (_B = 0; _B < batchSize; ++_B)
                {

                    for (_H = 0; _H < borderI; _H += strideH)
                    {

                        for (_W = 0; _W < borderJ; _W += strideW)
                        {

                            srcIndex = get_dst_index(_B, _CO, _H, _W, convC, dstW, dstH, strideH, strideW);

                            for (_J = 0; _J < convW; ++_J)
                            {

                                for (_I = 0; _I < convH; ++_I)
                                {

                                    __I = _H + _I;

                                    __J = _W + _J;

                                    dstIndex = get_src_index(_B, _CI, __I * featureW + __J, featureC, featureW, featureH);

                                    conIndex = get_con_index(_CO, _CI, _I, _J, featureC, convW, convH);

                                    dstMatrix[dstIndex] += (srcMatrix[srcIndex] * conMatrix[conIndex]);

                                }

                            }

                        }

                    }

                }

            }

        }

    }

    void transform_gradient_for_weights(
                            PARAM_TYPE * dstMatrix, // (batchSize, featureC, featureH, featureW)
                            PARAM_TYPE * srcMatrix, // (batchSize, convC, featureOutH, featureOutW)
                            PARAM_TYPE * conMatrix, // (convC, featureC, convH, convW)
                            size_t batchSize,
                            size_t featureH,
                            size_t featureW,
                            size_t featureC,
                            size_t convC,
                            size_t convW,
                            size_t convH,
                            size_t featureOutH,
                            size_t featureOutW,
                            size_t strideH,
                            size_t strideW
                            ) {

        size_t _CI, _CO, _B, _H, _W, _I, _J, __I, __J;

        size_t dstH = (featureH - convH) / strideH + 1;

        size_t dstW = (featureW - convW) / strideW + 1;

        size_t borderI = (featureH - convH) + 1;

        size_t borderJ = (featureW - convW) + 1;

        size_t srcIndex, dstIndex, conIndex;

        for (_CI = 0; _CI < featureC; ++_CI)
        {

            for (_CO = 0; _CO < convC; ++_CO)
            {

                for (_B = 0; _B < batchSize; ++_B)
                {

                    for (_H = 0; _H < borderI; _H += strideH)
                    {

                        for (_W = 0; _W < borderJ; _W += strideW)
                        {

                            srcIndex = get_dst_index(_B, _CO, _H, _W, convC, dstW, dstH, strideH, strideW);

                            for (_J = 0; _J < convW; ++_J)
                            {

                                for (_I = 0; _I < convH; ++_I)
                                {

                                    __I = _H + _I;

                                    __J = _W + _J;

                                    dstIndex = get_src_index(_B, _CI, __I * featureW + __J, featureC, featureW, featureH);

                                    conIndex = get_con_index(_CO, _CI, _I, _J, featureC, convW, convH);

                                    conMatrix[conIndex] += (srcMatrix[srcIndex] * dstMatrix[dstIndex]);

                                }

                            }

                        }

                    }


                    /*
                    for (_J = 0; _J < convW; ++_J)
                    {

                        for (_I = 0; _I < convH; ++_I)
                        {

                            conIndex = get_con_index(_CO, _CI, _I, _J, featureC, convW, convH);

                            for (_W = 0; _W < borderJ; _W += strideW)
                            {

                                for (_H = 0; _H < borderI; _H += strideH)
                                {

                                    __I = _H + _I;

                                    __J = _W + _J;

                                    dstIndex = get_src_index(_B, _CI, __I * featureW + __J, featureC, featureW, featureH);

                                    srcIndex = get_dst_index(_B, _CO, _H, _W, convC, dstW, dstH, strideH, strideW);

                                    conMatrix[conIndex] += (srcMatrix[srcIndex] * dstMatrix[dstIndex]);

                                }

                            }

                        }

                    }
                    */

                }

            }

        }

    }

}

#define H_CONVOLUTION

#endif // H_CONVOLUTION
