#define NDEBUG

#include "lstm.hpp"
#include "eigen_matrix.hpp"
#include "cuda_matrix.hpp"
#include "use_cuda.hpp"
#include <vector>
#include <pthread.h>
#include <atomic>
#include <iostream>
#include <cassert>
#include <queue>

std::atomic_int* progress3;

struct layerComparator
{
    // >= operation for a "max heap". We want a min-heap so we compare priorities with <=
  int operator()(const std::tuple<int, int>& tuple1,
                 const std::tuple<int, int>& tuple2)
  {
    return std::get<1>(tuple1) <= std::get<1>(tuple2);
  }
};

class PQ
{
    public:
        int get_layer_task();
        void add_layer_task(int layer, int priority);
        void check_max_priority(int priority);
        PQ(int max_priority);
    private:
        std::priority_queue<std::tuple<int, int>, std::vector<std::tuple<int, int>>, layerComparator> queue;
        int max_priority;
        bool finished;
        std::atomic_bool pq_busy;
};

PQ::PQ(int max_priority) {
   this->max_priority = max_priority;
   this->finished = false;
   this->pq_busy = false;
}

int PQ::get_layer_task(void) {
    bool success = false;
    // Wait for lock
    while (!success) {
        bool val = false;
        success = pq_busy.compare_exchange_weak(val, true);
    }
    if (finished) {
        pq_busy = false;
        return -2;
    }
    if (queue.empty()) {
        pq_busy = false;
        return -1;
    }
    std::tuple<int, int> tuple1 = queue.top(); queue.pop();
    int layer = std::get<0>(tuple1);
    int priority = std::get<1>(tuple1);
    check_max_priority(priority);
    pq_busy = false;
    return layer;
}

void PQ::add_layer_task(int layer, int priority) {
    bool success = false;
    // Wait for lock
    while (!success) {
        bool val = false;
        success = pq_busy.compare_exchange_weak(val, true);
    }
    queue.push(std::make_tuple(layer, priority));
    pq_busy = false;
}

void PQ::check_max_priority(int priority) {
    if(priority >= max_priority)
        finished = true;
}

template <class Matrix>
struct ThreadArgs {
    PQ* pq;
    std::vector<Matrix>* outputs;
    std::vector<Matrix>* hs;
    std::vector<Matrix>* cs;
    LSTM<Matrix>* lstm;
    int thread_idx;
};

template <class Matrix>
void* thread_fn3(void* args) {

    // Initialize Cublas
    if (use_cuda)
        cublas_init();

    struct ThreadArgs<Matrix>* thread_args = (struct ThreadArgs<Matrix>*) args;

    PQ* pq = thread_args->pq;
    std::vector<Matrix>* outputs = thread_args->outputs;
    std::vector<Matrix>* hs = thread_args->hs;
    std::vector<Matrix>* cs = thread_args->cs;
    LSTM<Matrix>* lstm = thread_args->lstm;
    int thread_idx = thread_args->thread_idx;

    int input_length = outputs->size();

    int layer_idx = -1;

    while (true) {
        while (layer_idx == -1){
            layer_idx = pq->get_layer_task();
        }
        if(layer_idx == -2)
            break;
        lstm->lstm_cells[layer_idx].forward((*outputs)[progress3[layer_idx].load(std::memory_order_relaxed)], (*hs)[layer_idx], (*cs)[layer_idx], (*hs)[layer_idx], (*cs)[layer_idx]);
        (*outputs)[progress3[layer_idx].load(std::memory_order_relaxed)] = (*hs)[layer_idx];
        int progress_old_layer = ++progress3[layer_idx];

        int old_layer = layer_idx;
        layer_idx = -1;
        int next_layer = old_layer + 1;
        if (next_layer < lstm->depth) {
            int progress_next_layer = progress3[next_layer].load(std::memory_order_relaxed);
            if (progress_next_layer + 1 >= progress_old_layer) {
                layer_idx = next_layer;
                pq->check_max_priority(next_layer + progress_next_layer);
            }
        }
        if (progress3[old_layer].load(std::memory_order_relaxed) < input_length) {
            int progress_below_layer = old_layer == 0 ? input_length : progress3[old_layer - 1].load(std::memory_order_relaxed);
            if (progress_below_layer >= progress_old_layer + 1) {
                int priority = old_layer + progress_old_layer;
                if (layer_idx == -1) {
                    layer_idx = old_layer;
                    pq->check_max_priority(priority);
                } else {
                    pq->add_layer_task(old_layer, priority);
                }
            }
        }
    }

    // Finalize Cublas
    if (use_cuda)
        cublas_finalize();

    return NULL;
}

template <class Matrix>
void LSTM<Matrix>::forward_par3(const std::vector<Matrix>& inputs, Matrix& output, int num_threads) {
    int input_length = inputs.size();

    progress3 = new std::atomic_int[depth];

    for (int i = 0; i < depth; i++) {
        progress3[i] = 0;
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

    //struct ThreadArgs<Matrix> args;
    struct ThreadArgs<Matrix> args[num_threads];
    pthread_t threads[num_threads];

    PQ pq = PQ(depth + input_length - 2);
    pq.add_layer_task(0, 0);

    //args.pq = &pq;
    // args.outputs = &outputs;
    // args.hs = &hs;
    // args.cs = &cs;
    // args.lstm = this;

    for (int thread_idx = 1; thread_idx < num_threads; thread_idx++) {
        args[thread_idx].pq = &pq;
        args[thread_idx].outputs = &outputs;
        args[thread_idx].hs = &hs;
        args[thread_idx].cs = &cs;
        args[thread_idx].lstm = this;
        args[thread_idx].thread_idx = thread_idx;
        pthread_create(&threads[thread_idx], NULL, thread_fn3<Matrix>, (void *)(&args[thread_idx]));
    }

    // for (int thread_idx = 1; thread_idx < num_threads; thread_idx++) {
    //     pthread_create(&threads[thread_idx], NULL, thread_fn3<Matrix>, (void *)(&args));
    // }

    args[0].pq = &pq;
    args[0].outputs = &outputs;
    args[0].hs = &hs;
    args[0].cs = &cs;
    args[0].lstm = this;
    args[0].thread_idx = 0;

    thread_fn3<Matrix>((void *)(&args[0]));

    for (int thread_idx = 1; thread_idx < num_threads; thread_idx++) {
        pthread_join(threads[thread_idx], NULL);
    }

    assert(hs[depth-1] == outputs[input_length-1]);
    output = hs[depth-1];

    // Free resources
    delete[] progress3;
}

template class LSTM<EigenMatrix>;
template class LSTM<CudaMatrix>;
