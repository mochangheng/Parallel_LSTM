#ifndef _NNLIB_
#define _NNLIB_

#include "eigen_matrix.hpp"
#include "cuda_matrix.hpp"

EigenMatrix operator+(double lhs, const EigenMatrix& rhs);
EigenMatrix operator-(double lhs, const EigenMatrix& rhs);
EigenMatrix operator*(double lhs, const EigenMatrix& rhs);

CudaMatrix operator+(double lhs, const CudaMatrix& rhs);
CudaMatrix operator-(double lhs, const CudaMatrix& rhs);
CudaMatrix operator*(double lhs, const CudaMatrix& rhs);

#endif
