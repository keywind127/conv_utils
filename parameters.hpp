#ifndef H_PARAMETERS

#include <cstdio>

#include <limits>

typedef float PARAM_TYPE;

constexpr auto MAX_PARAM_VALUE = std::numeric_limits<PARAM_TYPE>::max();

constexpr auto MIN_PARAM_VALUE = std::numeric_limits<PARAM_TYPE>::min();

#define H_PARAMETERS

#endif // H_PARAMETERS
