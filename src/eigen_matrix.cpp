#include <eigen_matrix.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <nnlib.hpp>
#include <iostream>
#include <cassert>

EigenMatrix::EigenMatrix(int row_dim, int col_dim) 
    : row_dim(row_dim)
    , col_dim(col_dim)
    , matrix(row_dim, col_dim) {}

EigenMatrix::EigenMatrix(Eigen::MatrixXd matrix)
    : row_dim(matrix.rows())
    , col_dim(matrix.cols())
    , matrix(matrix) {}

void EigenMatrix::_init_random() {
    matrix.setRandom();
}

void EigenMatrix::_init_zeros() {
    matrix.setZero();
}

const Eigen::MatrixXd& EigenMatrix::get_matrix() const {
    return matrix;
}

int EigenMatrix::rows() const {
    return row_dim;
}

int EigenMatrix::cols() const {
    return col_dim;
}

EigenMatrix EigenMatrix::operator*(const EigenMatrix& other) const {
    return EigenMatrix((matrix.array() * other.matrix.array()).matrix());
}

EigenMatrix EigenMatrix::operator*(double value) const {
    return EigenMatrix((matrix.array() * value).matrix());
}

EigenMatrix EigenMatrix::operator+(const EigenMatrix& other) const {
    return EigenMatrix(matrix + other.matrix);
}

EigenMatrix EigenMatrix::operator+(double value) const {
    return EigenMatrix((matrix.array() + value).matrix());
}

EigenMatrix EigenMatrix::operator-(const EigenMatrix& other) const {
    return EigenMatrix(matrix - other.matrix);
}

EigenMatrix EigenMatrix::operator-(double value) const {
    return EigenMatrix((matrix.array() - value).matrix());
}

EigenMatrix EigenMatrix::operator%(const EigenMatrix& other) const {
    if (matrix.cols() != other.matrix.rows()) {
        std::cout << "LHS: " << matrix.cols() << ", RHS: " << other.matrix.rows() << std::endl;
        assert(false);
    }
    return EigenMatrix(matrix * other.matrix);
}

bool EigenMatrix::operator==(const EigenMatrix& other) const {
    return matrix.isApprox(other.matrix);
}

EigenMatrix EigenMatrix::exp() const {
    return EigenMatrix(matrix.array().exp().matrix());
}

EigenMatrix EigenMatrix::sigmoid() const {
    auto arr = matrix.array();
    auto new_arr = 1 / (1 + (arr * -1).exp());
    return EigenMatrix(new_arr.matrix());
}

EigenMatrix EigenMatrix::tanh() const {
    auto arr = matrix.array();
    auto new_arr = ((arr * 2).exp() - 1) / ((arr * 2).exp() + 1);
    return EigenMatrix(new_arr.matrix());
}
