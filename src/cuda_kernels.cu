#include "cuda_kernels.hpp"
#include <iostream>

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

#define BLOCKSIZE 256

__global__ void multiply_kernel(const double *devA, const double *devB, double *devC, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        devC[tid] = devA[tid] * devB[tid];
    }
}

__global__ void multiply_scalar_kernel(const double *devA, double x, double *devC, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        devC[tid] = devA[tid] * x;
    }
}

__global__ void add_kernel(const double *devA, const double *devB, double *devC, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        devC[tid] = devA[tid] + devB[tid];
    }
}

__global__ void add_scalar_kernel(const double *devA, double x, double *devC, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        devC[tid] = devA[tid] + x;
    }
}

__global__ void subtract_kernel(const double *devA, const double *devB, double *devC, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        devC[tid] = devA[tid] - devB[tid];
    }
}

__global__ void subtract_scalar_kernel(const double *devA, double x, double *devC, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        devC[tid] = devA[tid] - x;
    }
}

__global__ void exp_kernel(const double *devA, double *devC, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        devC[tid] = exp(devA[tid]);
    }
}

__global__ void sigmoid_kernel(const double *devA, double *devC, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        devC[tid] = 1 / (1 + exp(-devA[tid]));
    }
}

__global__ void tanh_kernel(const double *devA, double *devC, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        devC[tid] = tanh(devA[tid]);
    }
}

void cudaMultiply(const double *devPtrA, const double *devPtrB, double *devPtrC, int numElem, cudaStream_t cuda_stream) {
    dim3 threadsPerBlock(BLOCKSIZE, 1);
    dim3 numBlocks((numElem + BLOCKSIZE - 1) / BLOCKSIZE, 1);
    multiply_kernel<<<numBlocks, threadsPerBlock, 0, cuda_stream>>>(devPtrA, devPtrB, devPtrC, numElem);
    
    cudaCheckError();
}

void cudaMultiplyScalar(const double *devPtrA, double x, double *devPtrC, int numElem, cudaStream_t cuda_stream) {
    dim3 threadsPerBlock(BLOCKSIZE, 1);
    dim3 numBlocks((numElem + BLOCKSIZE - 1) / BLOCKSIZE, 1);
    multiply_scalar_kernel<<<numBlocks, threadsPerBlock, 0, cuda_stream>>>(devPtrA, x, devPtrC, numElem);
    
    cudaCheckError();
}

void cudaAdd(const double *devPtrA, const double *devPtrB, double *devPtrC, int numElem, cudaStream_t cuda_stream) {
    dim3 threadsPerBlock(BLOCKSIZE, 1);
    dim3 numBlocks((numElem + BLOCKSIZE - 1) / BLOCKSIZE, 1);
    add_kernel<<<numBlocks, threadsPerBlock, 0, cuda_stream>>>(devPtrA, devPtrB, devPtrC, numElem);
    
    cudaCheckError();
}

void cudaAddScalar(const double *devPtrA, double x, double *devPtrC, int numElem, cudaStream_t cuda_stream) {
    dim3 threadsPerBlock(BLOCKSIZE, 1);
    dim3 numBlocks((numElem + BLOCKSIZE - 1) / BLOCKSIZE, 1);
    add_scalar_kernel<<<numBlocks, threadsPerBlock, 0, cuda_stream>>>(devPtrA, x, devPtrC, numElem);
    
    cudaCheckError();
}

void cudaSubtract(const double *devPtrA, const double *devPtrB, double *devPtrC, int numElem, cudaStream_t cuda_stream) {
    dim3 threadsPerBlock(BLOCKSIZE, 1);
    dim3 numBlocks((numElem + BLOCKSIZE - 1) / BLOCKSIZE, 1);
    subtract_kernel<<<numBlocks, threadsPerBlock, 0, cuda_stream>>>(devPtrA, devPtrB, devPtrC, numElem);
    
    cudaCheckError();
}

void cudaSubtractScalar(const double *devPtrA, double x, double *devPtrC, int numElem, cudaStream_t cuda_stream) {
    dim3 threadsPerBlock(BLOCKSIZE, 1);
    dim3 numBlocks((numElem + BLOCKSIZE - 1) / BLOCKSIZE, 1);
    subtract_scalar_kernel<<<numBlocks, threadsPerBlock, 0, cuda_stream>>>(devPtrA, x, devPtrC, numElem);
    
    cudaCheckError();
}

void cudaExp(const double *devPtrA, double *devPtrC, int numElem, cudaStream_t cuda_stream) {
    dim3 threadsPerBlock(BLOCKSIZE, 1);
    dim3 numBlocks((numElem + BLOCKSIZE - 1) / BLOCKSIZE, 1);
    exp_kernel<<<numBlocks, threadsPerBlock, 0, cuda_stream>>>(devPtrA, devPtrC, numElem);
    
    cudaCheckError();
}

void cudaSigmoid(const double *devPtrA, double *devPtrC, int numElem, cudaStream_t cuda_stream) {
    dim3 threadsPerBlock(BLOCKSIZE, 1);
    dim3 numBlocks((numElem + BLOCKSIZE - 1) / BLOCKSIZE, 1);
    sigmoid_kernel<<<numBlocks, threadsPerBlock, 0, cuda_stream>>>(devPtrA, devPtrC, numElem);
    
    cudaCheckError();
}

void cudaTanh(const double *devPtrA, double *devPtrC, int numElem, cudaStream_t cuda_stream) {
    dim3 threadsPerBlock(BLOCKSIZE, 1);
    dim3 numBlocks((numElem + BLOCKSIZE - 1) / BLOCKSIZE, 1);
    tanh_kernel<<<numBlocks, threadsPerBlock, 0, cuda_stream>>>(devPtrA, devPtrC, numElem);
    
    cudaCheckError();
}
