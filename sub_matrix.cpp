#include <bits/stdc++.h>

using namespace std;

typedef float PARAM_TYPE;

namespace SubMatrixExtraction
{

    inline size_t get_src_index(size_t h, size_t i, size_t j, size_t c, size_t featureH, size_t featureW, size_t featureC)
    {

        // h : batch_index
        // i : row index
        // j : col index
        // c : channel index

        size_t feature_WC = featureW * featureC;

        return c + j * featureC + i * feature_WC + h * featureH * feature_WC;

    }

    inline size_t get_dst_index(size_t h, size_t p, size_t dI, size_t dJ, size_t c, size_t partialN, size_t convH, size_t convW, size_t featureC)
    {

        // h  : batch index
        // p  : sub-matrix index
        // dI : row index
        // dJ : col index
        // c  : channel index

        size_t feature_WC = convW * featureC;

        size_t feature_HWC = convH * feature_WC;

        return c + dJ * featureC + dI * feature_WC + p * feature_HWC + h * partialN * feature_HWC;

    }

    void extract_sub_matrix(
                            PARAM_TYPE * dstMatrix, // (batchSize, partialN, convH, convW, featureC)
                            PARAM_TYPE * srcMatrix, // (batchSize, featureH, featureW, featureC)
                            size_t featureH,
                            size_t featureW,
                            size_t featureC,
                            size_t partialN,
                            size_t batchSize,
                            size_t convH,
                            size_t convW,
                            size_t strideH,
                            size_t strideW
                        ) {

        size_t i, j, k, l, m, n;

        //size_t dI, dJ;

        // 1D indexes for src and dst matrices
        size_t srcIndex, dstIndex;

        // exclusive border for row starting index
        size_t borderI = featureH - convH + 1;

        // exclusive border for col starting index
        size_t borderJ = featureW - convW + 1;

        // exclusive border for row index
        size_t borderK;

        // exclusive border for col index
        size_t borderL;

        // index of current sub-matrix
        size_t partialIndex = 0;

        // for each row starting index per vertical stride
        for (i = 0; i < borderI; i += strideH)
        {

            // row index of feature map
            //dI = i / strideH;

            // for each col starting index per horizontal stride
            for (j = 0; j < borderJ; j += strideW, partialIndex += 1)
            {

                // col index of feature map
                //dJ = j / strideW;

                // row index border
                borderK = i + convH;

                // iterate through row index
                for (k = i; k < borderK; ++k)
                {

                    // col index border
                    borderL = j + convW;

                    // iterate through col index
                    for (l = j; l < borderL; ++l)
                    {

                        // iterate through batch index
                        for (m = 0; m < batchSize; ++m)
                        {

                            // iterate through channel index
                            for (n = 0; n < featureC; ++n)
                            {

                                // current index of src matrix
                                srcIndex = get_src_index(m, k, l, n, featureH, featureW, featureC);

                                // current index of dst matrix
                                dstIndex = get_dst_index(m, partialIndex, (k - i), (l - j), n, partialN, convH, convW, featureC);

                                // copy value from src to dst matrix
                                dstMatrix[dstIndex] = srcMatrix[srcIndex];


                            }

                        }

                    }

                }

            }

        }

    }
}

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

extern "C"
{

    void extract_matrices(
                        PARAM_TYPE * dstMatrix, // (batchSize, partialN, convH, convW, featureC)
                        PARAM_TYPE * srcMatrix, // (batchSize, featureH, featureW, featureC)
                        size_t featureH,
                        size_t featureW,
                        size_t featureC,
                        size_t partialN,
                        size_t batchSize,
                        size_t convH,
                        size_t convW,
                        size_t strideH,
                        size_t strideW
                    ) {

        SubMatrixExtraction::extract_sub_matrix(dstMatrix, srcMatrix, featureH, featureW, featureC, partialN, batchSize, convH, convW, strideH, strideW);

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

        Convolution::apply_convolution(dstMatrix, srcMatrix, convVector, featureH, featureW, featureC, batchSize, convH, convW, convC, strideH, strideW);

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


        Convolution::transform_gradient(dstMatrix, srcMatrix, conMatrix, batchSize, featureH, featureW, featureC, convC, convW, convH, featureOutH, featureOutW, strideH, strideW);

    }

    void transform_gradient_for_weights(
                            PARAM_TYPE * srcMatrix, // (batchSize, featureC, featureH, featureW)
                            PARAM_TYPE * dstMatrix, // (batchSize, convC, featureOutH, featureOutW)
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

        Convolution::transform_gradient_for_weights(srcMatrix, dstMatrix, conMatrix, batchSize, featureH, featureW, featureC, convC, convW, convH, featureOutH, featureOutW, strideH, strideW);

    }

}

int main()
{

    return 0;
}
