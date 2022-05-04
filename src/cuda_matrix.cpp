#define NDEBUG

#include "cuda_matrix.hpp"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <Eigen/Dense>
#include "cuda_kernels.hpp"
#include <cassert>
#include <iostream>

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

thread_local bool has_stream = false;
thread_local cudaStream_t cuda_stream;
thread_local cublasHandle_t cublas_handle;

void cublas_init() {
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, cuda_stream);
    has_stream = true;
    cudaCheckError();
}

void cublas_finalize() {
    cublasDestroy(cublas_handle);
    cudaStreamDestroy(cuda_stream);
    has_stream = false;
    cudaCheckError();
}

bool approxEqual(double x, double y) {
    return std::abs(x-y) < 1e-5;
}

CudaMatrix::CudaMatrix(int row_dim, int col_dim)
    : row_dim(row_dim)
    , col_dim(col_dim) {
    cudaMalloc((void **)&cuda_data, row_dim * col_dim * sizeof(double));
}

CudaMatrix::CudaMatrix(int row_dim, int col_dim, double *cuda_data)
    : row_dim(row_dim)
    , col_dim(col_dim)
    , cuda_data(cuda_data) {}

CudaMatrix::CudaMatrix(CudaMatrix&& other) 
    : row_dim(other.row_dim)
    , col_dim(other.col_dim)
    , cuda_data(other.cuda_data) {
    other.cuda_data = nullptr;
}

CudaMatrix& CudaMatrix::operator=(CudaMatrix&& other) {
    if (cuda_data != nullptr) {
        cudaFree(cuda_data);
    }

    cuda_data = other.cuda_data;
    row_dim = other.row_dim;
    col_dim = other.col_dim;
    other.cuda_data = nullptr;
    return *this;
}

CudaMatrix::CudaMatrix(const CudaMatrix& other) 
    : row_dim(other.row_dim)
    , col_dim(other.col_dim) {
    cudaMalloc((void**)&cuda_data, other.row_dim * other.col_dim * sizeof(double));
    if (has_stream)
        cudaMemcpyAsync(cuda_data, other.cuda_data, other.row_dim * other.col_dim * sizeof(double), cudaMemcpyDeviceToDevice, cuda_stream);
    else
        cudaMemcpy(cuda_data, other.cuda_data, other.row_dim * other.col_dim * sizeof(double), cudaMemcpyDeviceToDevice);
}

CudaMatrix& CudaMatrix::operator=(const CudaMatrix& other) {
    if (cuda_data != nullptr) {
        cudaFree(cuda_data);
    }

    cudaMalloc((void**)&cuda_data, other.row_dim * other.col_dim * sizeof(double));
    if (has_stream)
        cudaMemcpyAsync(cuda_data, other.cuda_data, other.row_dim * other.col_dim * sizeof(double), cudaMemcpyDeviceToDevice, cuda_stream);
    else
        cudaMemcpy(cuda_data, other.cuda_data, other.row_dim * other.col_dim * sizeof(double), cudaMemcpyDeviceToDevice);
    row_dim = other.row_dim;
    col_dim = other.col_dim;
    return *this;
}

CudaMatrix::~CudaMatrix() {
    if (cuda_data != nullptr) {
        cudaFree(cuda_data);
    }
}

void CudaMatrix::_init_zeros() {
    if (has_stream)
        cudaMemsetAsync(cuda_data, 0, row_dim * col_dim * sizeof(double), cuda_stream);
    else
        cudaMemset(cuda_data, 0, row_dim * col_dim * sizeof(double));
}

void CudaMatrix::_init_random() {
    Eigen::MatrixXd eigen_matrix(row_dim, col_dim);
    eigen_matrix.setRandom();
    const double *host_data = eigen_matrix.data();
    if (has_stream)
        cudaMemcpyAsync(cuda_data, host_data, sizeof(double) * row_dim * col_dim, cudaMemcpyHostToDevice, cuda_stream);
    else
        cudaMemcpy(cuda_data, host_data, sizeof(double) * row_dim * col_dim, cudaMemcpyHostToDevice);
}

double *CudaMatrix::get_data() const {
    // NOTE: this gets data in column major format
    double *hostPtr = new double[row_dim * col_dim];
    if (has_stream)
        cudaMemcpyAsync(hostPtr, cuda_data, sizeof(double) * row_dim * col_dim, cudaMemcpyDeviceToHost, cuda_stream);
    else
        cudaMemcpy(hostPtr, cuda_data, sizeof(double) * row_dim * col_dim, cudaMemcpyDeviceToHost);
    return hostPtr;
}

int CudaMatrix::rows() const {
    return row_dim;
}

int CudaMatrix::cols() const {
    return col_dim;
}

CudaMatrix CudaMatrix::operator*(const CudaMatrix& other) const {
    assert(has_stream);
    double *out_data;
    cudaMalloc((void**)&out_data, row_dim * col_dim * sizeof(double));
    cudaMultiply(cuda_data, other.cuda_data, out_data, row_dim * col_dim, cuda_stream);
    cudaCheckError();
    return CudaMatrix(row_dim, col_dim, out_data);
}

CudaMatrix CudaMatrix::operator*(double value) const {
    assert(has_stream);
    double *out_data;
    cudaMalloc((void**)&out_data, row_dim * col_dim * sizeof(double));
    cudaMultiplyScalar(cuda_data, value, out_data, row_dim * col_dim, cuda_stream);
    cudaCheckError();
    return CudaMatrix(row_dim, col_dim, out_data);
}

CudaMatrix CudaMatrix::operator+(const CudaMatrix& other) const {
    assert(has_stream);
    double *out_data;
    cudaMalloc((void**)&out_data, row_dim * col_dim * sizeof(double));
    cudaAdd(cuda_data, other.cuda_data, out_data, row_dim * col_dim, cuda_stream);
    cudaCheckError();
    return CudaMatrix(row_dim, col_dim, out_data);
}

CudaMatrix CudaMatrix::operator+(double value) const {
    assert(has_stream);
    double *out_data;
    cudaMalloc((void**)&out_data, row_dim * col_dim * sizeof(double));
    cudaAddScalar(cuda_data, value, out_data, row_dim * col_dim, cuda_stream);
    cudaCheckError();
    return CudaMatrix(row_dim, col_dim, out_data);
}

CudaMatrix CudaMatrix::operator-(const CudaMatrix& other) const {
    assert(has_stream);
    double *out_data;
    cudaMalloc((void**)&out_data, row_dim * col_dim * sizeof(double));
    cudaSubtract(cuda_data, other.cuda_data, out_data, row_dim * col_dim, cuda_stream);
    cudaCheckError();
    return CudaMatrix(row_dim, col_dim, out_data);
}

CudaMatrix CudaMatrix::operator-(double value) const {
    assert(has_stream);
    double *out_data;
    cudaMalloc((void**)&out_data, row_dim * col_dim * sizeof(double));
    cudaSubtractScalar(cuda_data, value, out_data, row_dim * col_dim, cuda_stream);
    cudaCheckError();
    return CudaMatrix(row_dim, col_dim, out_data);
}

CudaMatrix CudaMatrix::operator%(const CudaMatrix& other) const {
    assert(has_stream);
    assert(col_dim == other.row_dim);
    double *out_data;
    cudaMalloc((void**)&out_data, row_dim * other.col_dim * sizeof(double));
    double alpha = 1.0;
    double beta = 0.0;
    if (other.col_dim == 1) {
        cublasDgemv(cublas_handle, CUBLAS_OP_N, row_dim, col_dim, &alpha, cuda_data, row_dim, other.cuda_data, 1, &beta, out_data, 1);
    } else {
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, row_dim, other.col_dim, col_dim, &alpha, cuda_data, row_dim, other.cuda_data, other.row_dim, &beta, out_data, row_dim);
    }
    cudaCheckError();
    return CudaMatrix(row_dim, other.col_dim, out_data);
}

bool CudaMatrix::operator==(const CudaMatrix& other) const {
    int num_elem = row_dim * col_dim;
    double *data = get_data();
    double *other_data = other.get_data();

    bool is_equal = true;
    for (int i = 0; i < num_elem; i++) {
        if (!approxEqual(data[i], other_data[i])) {
            is_equal = false;
            break;
        }
    }

    delete[] data;
    delete[] other_data;
    return is_equal;
}

CudaMatrix CudaMatrix::exp() const {
    assert(has_stream);
    double *out_data;
    cudaMalloc((void**)&out_data, row_dim * col_dim * sizeof(double));
    cudaExp(cuda_data, out_data, row_dim * col_dim, cuda_stream);
    cudaCheckError();
    return CudaMatrix(row_dim, col_dim, out_data);
}

CudaMatrix CudaMatrix::sigmoid() const {
    assert(has_stream);
    double *out_data;
    cudaMalloc((void**)&out_data, row_dim * col_dim * sizeof(double));
    cudaSigmoid(cuda_data, out_data, row_dim * col_dim, cuda_stream);
    cudaCheckError();
    return CudaMatrix(row_dim, col_dim, out_data);
}

CudaMatrix CudaMatrix::tanh() const {
    assert(has_stream);
    double *out_data;
    cudaMalloc((void**)&out_data, row_dim * col_dim * sizeof(double));
    cudaTanh(cuda_data, out_data, row_dim * col_dim, cuda_stream);
    cudaCheckError();
    return CudaMatrix(row_dim, col_dim, out_data);
}
