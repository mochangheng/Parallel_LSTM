void cudaMultiply(const double *devPtrA, const double *devPtrB, double *devPtrC, int numElem, cudaStream_t cuda_stream);
void cudaMultiplyScalar(const double *devPtrA, double x, double *devPtrC, int numElem, cudaStream_t cuda_stream);

void cudaAdd(const double *devPtrA, const double *devPtrB, double *devPtrC, int numElem, cudaStream_t cuda_stream);
void cudaAddScalar(const double *devPtrA, double x, double *devPtrC, int numElem, cudaStream_t cuda_stream);

void cudaSubtract(const double *devPtrA, const double *devPtrB, double *devPtrC, int numElem, cudaStream_t cuda_stream);
void cudaSubtractScalar(const double *devPtrA, double x, double *devPtrC, int numElem, cudaStream_t cuda_stream);

void cudaExp(const double *devPtrA, double *devPtrC, int numElem, cudaStream_t cuda_stream);
void cudaSigmoid(const double *devPtrA, double *devPtrC, int numElem, cudaStream_t cuda_stream);
void cudaTanh(const double *devPtrA, double *devPtrC, int numElem, cudaStream_t cuda_stream);
