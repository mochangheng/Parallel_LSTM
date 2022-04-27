#include "nnlib.hpp"
#include "eigen_matrix.hpp"

EigenMatrix operator+(double lhs, const EigenMatrix& rhs) {
    return rhs + lhs;
}

EigenMatrix operator-(double lhs, const EigenMatrix& rhs) {
    return rhs * (-1.0f) + lhs;
}

EigenMatrix operator*(double lhs, const EigenMatrix& rhs) {
    return rhs * lhs;
}

CudaMatrix operator+(double lhs, const CudaMatrix& rhs) {
    return rhs + lhs;
}

CudaMatrix operator-(double lhs, const CudaMatrix& rhs) {
    return rhs * (-1.0f) + lhs;
}

CudaMatrix operator*(double lhs, const CudaMatrix& rhs) {
    return rhs * lhs;
}
