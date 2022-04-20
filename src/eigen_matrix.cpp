#include <eigen_matrix.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <nnlib.hpp>
#include <iostream>
#include <cassert>
#include <stdexcept>

/* Broadcast two matrices such that their shapes match, if possible */
void broadcast_matrices(const Eigen::MatrixXd& m1, const Eigen::MatrixXd& m2, Eigen::MatrixXd& out1, Eigen::MatrixXd& out2) {
    if (m1.rows() != m2.rows()) {
        if (m1.cols() != m2.cols()) {
            throw std::invalid_argument("Cannot be broadcasted");
        }
        if (m1.rows() == 1) {
            out1 = m1.replicate(m2.rows(), 1);
            out2 = m2;
        } else if (m2.rows() == 1) {
            out2 = m2.replicate(m1.rows(), 1);
            out1 = m1;
        } else {
            throw std::invalid_argument("Cannot be broadcasted");
        }
    } else if (m1.cols() != m2.cols()) {
        if (m1.cols() == 1) {
            out1 = m1.replicate(1, m2.cols());
            out2 = m2;
        } else if (m2.cols() == 1) {
            out2 = m2.replicate(1, m1.cols());
            out1 = m1;
        } else {
            throw std::invalid_argument("Cannot be broadcasted");
        }
    }
}

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
    Eigen::MatrixXd m1, m2;
    // Broadcasting if needed
    if ((matrix.rows() != other.matrix.rows()) || (matrix.cols() != other.matrix.cols())) {
        broadcast_matrices(matrix, other.matrix, m1, m2);
    } else {
        m1 = matrix;
        m2 = other.matrix;
    }
    return EigenMatrix((m1.array() * m2.array()).matrix());
}

EigenMatrix EigenMatrix::operator*(double value) const {
    return EigenMatrix((matrix.array() * value).matrix());
}

EigenMatrix EigenMatrix::operator+(const EigenMatrix& other) const {
    Eigen::MatrixXd m1, m2;
    // Broadcasting if needed
    if ((matrix.rows() != other.matrix.rows()) || (matrix.cols() != other.matrix.cols())) {
        broadcast_matrices(matrix, other.matrix, m1, m2);
    } else {
        m1 = matrix;
        m2 = other.matrix;
    }
    return EigenMatrix(m1 + m2);
}

EigenMatrix EigenMatrix::operator+(double value) const {
    return EigenMatrix((matrix.array() + value).matrix());
}

EigenMatrix EigenMatrix::operator-(const EigenMatrix& other) const {
    Eigen::MatrixXd m1, m2;
    // Broadcasting if needed
    if ((matrix.rows() != other.matrix.rows()) || (matrix.cols() != other.matrix.cols())) {
        broadcast_matrices(matrix, other.matrix, m1, m2);
    } else {
        m1 = matrix;
        m2 = other.matrix;
    }
    return EigenMatrix(m1 - m2);
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
