#ifndef EIGEN_TENSOR
#define EIGEN_TENSOR

#include <Eigen/Dense>

class EigenMatrix {
private:
    int row_dim;
    int col_dim;
    Eigen::MatrixXd matrix;

public:
    EigenMatrix(int row_dim, int col_dim);
    EigenMatrix(Eigen::MatrixXd matrix);

    void _init_random();
    void _init_zeros();
    const Eigen::MatrixXd& get_matrix() const;
    int rows() const;
    int cols() const;

    // Binary operations
    EigenMatrix operator*(const EigenMatrix& other) const; // Elementwise
    EigenMatrix operator*(double value) const;
    EigenMatrix operator+(const EigenMatrix& other) const;
    EigenMatrix operator+(double value) const;
    EigenMatrix operator-(const EigenMatrix& other) const;
    EigenMatrix operator-(double value) const;
    EigenMatrix operator%(const EigenMatrix& other) const; // Matrix multiplication
    bool operator==(const EigenMatrix& other) const;

    // Unary operations
    EigenMatrix exp() const;
    EigenMatrix sigmoid() const;
    EigenMatrix tanh() const;
};

#endif
