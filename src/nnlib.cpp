#include <nnlib.hpp>
#include <eigen_matrix.hpp>

EigenMatrix operator+(double lhs, const EigenMatrix& rhs) {
    return rhs + lhs;
}

EigenMatrix operator-(double lhs, const EigenMatrix& rhs) {
    return -1 * rhs + lhs;
}

EigenMatrix operator*(double lhs, const EigenMatrix& rhs) {
    return rhs * lhs;
}
