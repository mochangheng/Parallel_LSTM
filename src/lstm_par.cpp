#include <lstm.hpp>
#include <eigen_matrix.hpp>
#include <vector>
#include <pthread.h>
#include <atomic>
#include <iostream>
#include <cassert>

std::atomic_bool* busy;
int* progress;

template <class Matrix>
struct ThreadArgs {
    std::vector<Matrix>* outputs;
    std::vector<Matrix>* hs;
    std::vector<Matrix>* cs;
    LSTM<Matrix>* lstm;
    int thread_id;
};

template <class Matrix>
void* thread_fn(void* args) {
    struct ThreadArgs<Matrix>* thread_args = (struct ThreadArgs<Matrix>*) args;

    std::vector<Matrix>* outputs = thread_args->outputs;
    std::vector<Matrix>* hs = thread_args->hs;
    std::vector<Matrix>* cs = thread_args->cs;
    LSTM<Matrix>* lstm = thread_args->lstm;

    int input_length = outputs->size();
    int depth = lstm->depth;

    while (true) {
        bool all_done = true;

        for (int i = 0; i < input_length; i++) {
            int prev_progress = (i == 0) ? depth : progress[i-1];
            if (prev_progress > progress[i] && !busy[i]) {
                // NOTE: we don't need this condition but this might reduce contention
                bool val = false;
                bool success = busy[i].compare_exchange_weak(val, true);
                if (success) { // Acquired the lock
                    // Update `prev_progress` to get newest information
                    prev_progress = (i == 0) ? depth : progress[i-1];
                    if (prev_progress > progress[i]) {
                        // Do a forward pass
                        int depth_idx = progress[i];
                        lstm->lstm_cells[depth_idx].forward((*outputs)[i], (*hs)[depth_idx], (*cs)[depth_idx], (*hs)[depth_idx], (*cs)[depth_idx]);
                        (*outputs)[i] = (*hs)[depth_idx];
                        progress[i]++;
                    }
                    // Release the lock
                    busy[i] = false;
                }
            }

            if (progress[i] < depth)
                all_done = false;

            if (progress[i] == 0)
                break;
        }

        if (all_done)
            break;
    }

    return NULL;
}

template <class Matrix>
void LSTM<Matrix>::forward_par1(const std::vector<Matrix>& inputs, Matrix& output, int num_threads) {
    int input_length = inputs.size();

    busy = new std::atomic_bool[input_length];
    progress = new int[input_length];
    for (int i = 0; i < input_length; i++) {
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

    struct ThreadArgs<Matrix> args[num_threads];
    pthread_t threads[num_threads];

    for (int thread_idx = 1; thread_idx < num_threads; thread_idx++) {
        args[thread_idx].outputs = &outputs;
        args[thread_idx].hs = &hs;
        args[thread_idx].cs = &cs;
        args[thread_idx].lstm = this;
        args[thread_idx].thread_id = thread_idx;
        pthread_create(&threads[thread_idx], NULL, thread_fn<Matrix>, (void *)(&args[thread_idx]));
    }

    args[0].outputs = &outputs;
    args[0].hs = &hs;
    args[0].cs = &cs;
    args[0].lstm = this;
    args[0].thread_id = 0;
    thread_fn<Matrix>((void *)(&args[0]));

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
