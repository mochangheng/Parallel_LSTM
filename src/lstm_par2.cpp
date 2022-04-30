#define NDEBUG

#include "lstm.hpp"
#include "eigen_matrix.hpp"
#include "cuda_matrix.hpp"
#include <vector>
#include <pthread.h>
#include <atomic>
#include <iostream>
#include <cassert>

// Set this flag to use cuda
const bool use_cuda = false;
int* progress2; // Somehow `progress` conflicts with `lstm_par.cpp`
std::atomic_int next_layer = 0;

template <class Matrix>
struct ThreadArgs {
    std::vector<Matrix>* outputs;
    std::vector<Matrix>* hs;
    std::vector<Matrix>* cs;
    LSTM<Matrix>* lstm;
};

template <class Matrix>
void* thread_fn2(void* args) {

    // Initialize Cublas
    if (use_cuda)
        cublas_init();

    struct ThreadArgs<Matrix>* thread_args = (struct ThreadArgs<Matrix>*) args;

    std::vector<Matrix>* outputs = thread_args->outputs;
    std::vector<Matrix>* hs = thread_args->hs;
    std::vector<Matrix>* cs = thread_args->cs;
    LSTM<Matrix>* lstm = thread_args->lstm;

    int input_length = outputs->size();

    while (true) {
        int layer_idx = next_layer++;
        if (layer_idx >= lstm->depth)
            break;
        
        for (int input_idx = 0; input_idx < input_length; input_idx++) {
            if (layer_idx > 0)
                while (progress2[layer_idx-1] <= input_idx);

            lstm->lstm_cells[layer_idx].forward((*outputs)[input_idx], (*hs)[layer_idx], (*cs)[layer_idx], (*hs)[layer_idx], (*cs)[layer_idx]);
            (*outputs)[input_idx] = (*hs)[layer_idx];
            progress2[layer_idx]++;
        }
    }

    // Finalize Cublas
    if (use_cuda)
        cublas_finalize();

    return NULL;
}

template <class Matrix>
void LSTM<Matrix>::forward_par2(const std::vector<Matrix>& inputs, Matrix& output, int num_threads) {
    int input_length = inputs.size();

    progress2 = new int[depth];

    for (int i = 0; i < depth; i++) {
        progress2[i] = 0;
    }

    std::vector<Matrix> hs;
    std::vector<Matrix> cs;
    std::vector<Matrix> outputs(inputs);

    for (int i = 0; i < depth; i++) {
        Matrix h(hidden_size, 1), c(hidden_size, 1);
        h._init_zeros();
        c._init_zeros();
        hs.push_back(h);
        cs.push_back(c);
    }

    struct ThreadArgs<Matrix> args;
    pthread_t threads[num_threads];

    args.outputs = &outputs;
    args.hs = &hs;
    args.cs = &cs;
    args.lstm = this;

    for (int thread_idx = 1; thread_idx < num_threads; thread_idx++) {
        pthread_create(&threads[thread_idx], NULL, thread_fn2<Matrix>, (void *)(&args));
    }

    thread_fn2<Matrix>((void *)(&args));

    for (int thread_idx = 1; thread_idx < num_threads; thread_idx++) {
        pthread_join(threads[thread_idx], NULL);
    }

    assert(hs[depth-1] == outputs[input_length-1]);
    output = hs[depth-1];

    // Free resources
    delete[] progress2;
}

template class LSTM<EigenMatrix>;
template class LSTM<CudaMatrix>;
