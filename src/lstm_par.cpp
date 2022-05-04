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
const bool use_cuda = true;
int* progress;
std::atomic_bool* busy;

template <class Matrix>
struct ThreadArgs {
    std::vector<Matrix>* outputs;
    std::vector<Matrix>* hs;
    std::vector<Matrix>* cs;
    LSTM<Matrix>* lstm;
};

template <class Matrix>
void* thread_fn(void* args) {

    // Initialize Cublas
    if (use_cuda)
        cublas_init();

    struct ThreadArgs<Matrix>* thread_args = (struct ThreadArgs<Matrix>*) args;

    std::vector<Matrix>* outputs = thread_args->outputs;
    std::vector<Matrix>* hs = thread_args->hs;
    std::vector<Matrix>* cs = thread_args->cs;
    LSTM<Matrix>* lstm = thread_args->lstm;

    int input_length = outputs->size();
    int depth = lstm->depth;

    while (true) {
        bool all_done = true;

        for (int i = 0; i < depth; i++) {
            bool updated = false;
            int prev_progress = (i == 0) ? input_length : progress[i-1];
            if (prev_progress > progress[i] && !busy[i]) {
                // NOTE: Check execution conditions before trying to get lock to reduce contention
                bool val = false;
                bool success = busy[i].compare_exchange_weak(val, true);
                if (success) { // Acquired the lock
                    // Check execution conditions again
                    prev_progress = (i == 0) ? input_length : progress[i-1];
                    if (prev_progress > progress[i]) {
                        // If conditions satisfied, do a forward pass
                        int input_idx = progress[i];
                        lstm->lstm_cells[i].forward((*outputs)[input_idx], (*hs)[i], (*cs)[i], (*hs)[i], (*cs)[i]);
                        (*outputs)[input_idx] = (*hs)[i];
                        progress[i]++;
                        updated = true;
                    }
                    // Release the lock
                    busy[i] = false;
                }
            }

            if (progress[i] < input_length)
                all_done = false;

            if (progress[i] == 0)
                break;

            if (updated) // If we do a forward pass, stay at the same layer for better locality
                i--;
        }

        if (all_done)
            break;
    }

    // Finalize Cublas
    if (use_cuda)
        cublas_finalize();

    return NULL;
}

template <class Matrix>
void LSTM<Matrix>::forward_par1(const std::vector<Matrix>& inputs, Matrix& output, int num_threads) {
    int input_length = inputs.size();

    busy = new std::atomic_bool[depth];
    progress = new int[depth];
    for (int i = 0; i < depth; i++) {
        busy[i] = false;
        progress[i] = 0;
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
        pthread_create(&threads[thread_idx], NULL, thread_fn<Matrix>, (void *)(&args));
    }

    thread_fn<Matrix>((void *)(&args));

    for (int thread_idx = 1; thread_idx < num_threads; thread_idx++) {
        pthread_join(threads[thread_idx], NULL);
    }

    assert(hs[depth-1] == outputs[input_length-1]);
    output = hs[depth-1];

    // Free resources
    delete[] busy;
    delete[] progress;
}

template class LSTM<EigenMatrix>;
template class LSTM<CudaMatrix>;
