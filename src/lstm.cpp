#include "lstm.hpp"
#include "eigen_matrix.hpp"
#include "cuda_matrix.hpp"
#include <iostream>
#include <vector>

template <class Matrix>
LSTMCell<Matrix>::LSTMCell(int input_size, int hidden_size)
    : Wii(hidden_size, input_size)
    , bii(hidden_size, 1)
    , Whi(hidden_size, hidden_size)
    , bhi(hidden_size, 1)
    , Wif(hidden_size, input_size)
    , bif(hidden_size, 1)
    , Whf(hidden_size, hidden_size)
    , bhf(hidden_size, 1)
    , Wig(hidden_size, input_size)
    , big(hidden_size, 1)
    , Whg(hidden_size, hidden_size)
    , bhg(hidden_size, 1)
    , Wio(hidden_size, input_size)
    , bio(hidden_size, 1)
    , Who(hidden_size, hidden_size)
    , bho(hidden_size, 1) {
    // Initialize random weights and zero bias
    Wii._init_random();
    bii._init_zeros();
    Whi._init_random();
    bhi._init_zeros();
    Wif._init_random();
    bif._init_zeros();
    Whf._init_random();
    bhf._init_zeros();
    Wig._init_random();
    big._init_zeros();
    Whg._init_random();
    bhg._init_zeros();
    Wio._init_random();
    bio._init_zeros();
    Who._init_random();
    bho._init_zeros();
}

template <class Matrix>
void LSTMCell<Matrix>::forward(const Matrix& x, const Matrix& h, const Matrix& c, Matrix& h_out, Matrix& c_out) {
    auto it = (Wii % x + bii + Whi % h + bhi).sigmoid();
    auto ft = (Wif % x + bif + Whf % h + bhf).sigmoid();
    auto gt = (Wig % x + big + Whg % h + bhg).tanh();
    auto ot = (Wio % x + bio + Who % h + bho).sigmoid();
    c_out = (ft * c) + (it * gt);
    h_out = ot * c_out.tanh();
}

template <class Matrix>
LSTM<Matrix>::LSTM(int input_size, int hidden_size, int depth)
    : input_size(input_size)
    , hidden_size(hidden_size)
    , depth(depth) {
    for (int i = 0; i < depth; i++) {
        if (i == 0) {
            lstm_cells.push_back(LSTMCell<Matrix>(input_size, hidden_size));
        }
        else {
            lstm_cells.push_back(LSTMCell<Matrix>(hidden_size, hidden_size));
        }
    }
}

template <class Matrix>
void LSTM<Matrix>::forward(const std::vector<Matrix>& inputs, Matrix& output) {

    cublas_init();

    int input_length = inputs.size();

    std::vector<Matrix> hs;
    std::vector<Matrix> cs;

    for (int d = 0; d < depth; d++) {
        Matrix h(hidden_size, 1);
        Matrix c(hidden_size, 1);
        h._init_zeros();
        c._init_zeros();
        hs.push_back(h);
        cs.push_back(c);
    }

    for (int t = 0; t < input_length; t++) {
        Matrix x = inputs[t];
        for (int d = 0; d < depth; d++) {
            lstm_cells[d].forward(x, hs[d], cs[d], hs[d], cs[d]);
            x = hs[d];
        }
    }

    output = hs[depth-1];

    cublas_finalize();

}

template class LSTMCell<EigenMatrix>;
template class LSTM<EigenMatrix>;
template class LSTMCell<CudaMatrix>;
template class LSTM<CudaMatrix>;
