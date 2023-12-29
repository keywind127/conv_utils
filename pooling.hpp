#ifndef H_POOLING

#include <cstdio>

#include "parameters.hpp"

namespace Pooling
{

    inline size_t get_matrix_index(
                                size_t batchIndex,
                                size_t channelIndex,
                                size_t heightIndex,
                                size_t widthIndex,
                                size_t featureW,
                                size_t featureH,
                                size_t featureC) {

        size_t featureHW = featureH * featureW;

        return widthIndex + heightIndex * featureW + channelIndex * featureHW + batchIndex * featureC * featureHW;

    }

    void max_pooling_2d(
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

        size_t _B, _C, _H, _W, i, j;

        size_t rowBorder = featureH - poolH + 1;

        size_t colBorder = featureW - poolW + 1;

        PARAM_TYPE maxValue, tmpValue;

        size_t maxIndex, tmpIndex;

        for (_B = 0; _B < batchSize; ++_B)
        {

            for (_C = 0; _C < featureC; ++_C)
            {

                for (_H = 0; _H < rowBorder; _H += poolH)
                {

                    for (_W = 0; _W < colBorder; _W += poolW)
                    {

                        maxValue = MIN_PARAM_VALUE;

                        for (i = 0; i < poolH; ++i)
                        {

                            for (j = 0; j < poolW; ++j)
                            {

                                tmpIndex = get_matrix_index(_B, _C, _H + i, _W + j, featureW, featureH, featureC);

                                tmpValue = srcMatrix[tmpIndex];

                                if (tmpValue >= maxValue)
                                {

                                    maxValue = tmpValue;

                                    maxIndex = tmpIndex;

                                }

                            }

                        }

                        binMatrix[maxIndex] = 1;

                        dstMatrix[get_matrix_index(_B, _C, _H / poolH, _W / poolW, featureOutW, featureOutH, featureC)] = maxValue;

                    }

                }

            }

        }

    }

}

#define H_POOLING

#endif // H_POOLING
