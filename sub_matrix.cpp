#include "parameters.hpp"

#include "submatrix_extraction.hpp"

#include "convolution.hpp"

#include "pooling.hpp"

#include <cstdio>

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

    void apply_convolution_2(
                             PARAM_TYPE * dstMatrix, // (BS, OC, OH, OW)
                             PARAM_TYPE * srcMatrix, // (BS, IC, IH, IW)
                             PARAM_TYPE * conMatrix, // (OC, IC, CH, CW)
                             size_t imageHeight,
                             size_t imageWidth,
                             size_t inChannels,
                             size_t batchSize,
                             size_t convHeight,
                             size_t convWidth,
                             size_t outChannels,
                             size_t strideHeight,
                             size_t strideWidth) {

        Convolution::apply_convolution_2(dstMatrix, srcMatrix, conMatrix, batchSize, outChannels, inChannels, imageHeight, imageWidth, convHeight, convWidth, strideHeight, strideWidth);

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

    void max_pooling_2D(
                        PARAM_TYPE * dstMatrix,
                        PARAM_TYPE * binMatrix,
                        PARAM_TYPE * srcMatrix,
                        size_t batchSize,
                        size_t featureC,
                        size_t featureH,
                        size_t featureW,
                        size_t featureOutH,
                        size_t featureOutW,
                        size_t poolH,
                        size_t poolW
                        ) {

        Pooling::max_pooling_2d(dstMatrix, binMatrix, srcMatrix, batchSize, featureC, featureH, featureW, featureOutH, featureOutW, poolH, poolW);

    }

}

int main()
{

    return 0;
}
