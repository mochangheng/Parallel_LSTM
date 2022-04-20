#ifndef _LSTM_
#define _LSTM_

#include <vector>

template <class Matrix>
class LSTMCell {
private:
    Matrix Wii;
    Matrix bii;
    Matrix Whi;
    Matrix bhi;
    Matrix Wif;
    Matrix bif;
    Matrix Whf;
    Matrix bhf;
    Matrix Wig;
    Matrix big;
    Matrix Whg;
    Matrix bhg;
    Matrix Wio;
    Matrix bio;
    Matrix Who;
    Matrix bho;

public:
    LSTMCell(int input_size, int hidden_size);
    void forward(const Matrix& x, const Matrix& h, const Matrix& c, Matrix& h_out, Matrix& c_out);
};

template <class Matrix>
class LSTM {
private:
    int input_size;
    int hidden_size;
    int depth;
    std::vector<LSTMCell<Matrix>> lstm_cells;

public:
    LSTM(int input_size, int hidden_size, int depth);
    void forward(const std::vector<Matrix>& inputs, Matrix& output);
    void forward_par1(const std::vector<Matrix>& inputs, Matrix& output, int num_threads = 1);

    // Friend thread function
    template <class T>
    friend void* thread_fn (void* args);
};

#endif
