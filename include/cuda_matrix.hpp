#ifndef _CUDA_TENSOR_
#define _CUDA_TENSOR_

#include <cuda_runtime.h>
#include "cublas_v2.h"

void cublas_init();
void cublas_finalize();

class CudaMatrix {
private:
    int row_dim;
    int col_dim;
    double* cuda_data;

public:
    CudaMatrix(int row_dim, int col_dim);
    CudaMatrix(int row_dim, int col_dim, double *cuda_data);
    CudaMatrix(CudaMatrix&& other); // Move constructor
    CudaMatrix& operator=(CudaMatrix&& other); // Move assignment
    CudaMatrix(const CudaMatrix& other); // Copy constructor
    CudaMatrix& operator=(const CudaMatrix& other); // Copy assignment
    ~CudaMatrix();

    void _init_random();
    void _init_zeros();
    double *get_data() const;
    int rows() const;
    int cols() const;

    // Binary operations
    CudaMatrix operator*(const CudaMatrix& other) const; // Elementwise
    CudaMatrix operator*(double value) const;
    CudaMatrix operator+(const CudaMatrix& other) const;
    CudaMatrix operator+(double value) const;
    CudaMatrix operator-(const CudaMatrix& other) const;
    CudaMatrix operator-(double value) const;
    CudaMatrix operator%(const CudaMatrix& other) const; // Matrix multiplication
    bool operator==(const CudaMatrix& other) const;

    // Unary operations
    CudaMatrix exp() const;
    CudaMatrix sigmoid() const;
    CudaMatrix tanh() const;
};

#endif
