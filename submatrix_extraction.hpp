#ifndef H_SUBMATRIX_EXTRACTION

#include "parameters.hpp"

#include <cstdio>

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

#define H_SUBMATRIX_EXTRACTION

#endif // H_SUBMATRIX_EXTRACTION
