#define NDEBUG

#include <iostream>
#include <Eigen/Dense>
#include "eigen_matrix.hpp"
#include "cuda_matrix.hpp"
#include "lstm.hpp"
#include <cmath>
#include <vector>
#include <chrono>
#include <getopt.h>
#include <omp.h>
#include <cassert>

using Eigen::MatrixXd;
using namespace std::chrono;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> dsec;

bool arrayEqual(double *a, double *b, int n) {
  for (int i = 0; i < n; i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

void arrayAdd(double *a, double *b, double *out, int n) {
  for (int i = 0; i < n; i++) {
    out[i] = a[i] + b[i];
  }
}

void arrayPrint(double *a, int n) {
  for (int i = 0; i < n; i++) {
    std::cout << a[i] << std::endl;
  }
}

void cudaTest() {
  CudaMatrix m1(128, 128);
  CudaMatrix m2(128, 128);
  m1._init_random();
  m2._init_random();

  int num_elem = m1.rows() * m1.cols();

  double *m1_data = m1.get_data();
  double *m2_data = m2.get_data();

  CudaMatrix m3 = m1 + m2;
  double *m3_data = m3.get_data();

  assert(m3.rows() == m1.rows() && m3.cols() == m1.cols());
  
  arrayAdd(m1_data, m2_data, m1_data, num_elem);
  assert(arrayEqual(m1_data, m3_data, num_elem));
  
  delete[] m1_data;
  delete[] m2_data;
  delete[] m3_data;
}

int main(int argc, char* argv[]) {
  // Constants
  int do_par = 1;
  bool check_correct = false;

  // Cmd line arguments
  bool use_cuda = false;
  int depth = 4;
  int hidden_size = 128;
  int timesteps = 128;
  int batch_size = 1;

  int num_threads = 1;
  int matrix_threads = 1;
  int c;

  while ((c = getopt(argc, argv, "d:h:t:b:n:m:c")) != -1) {
    switch (c) {
    case 'd':
      depth = atoi(optarg);
      break;

    case 'h':
      hidden_size = atoi(optarg);
      break;

    case 't':
      timesteps = atoi(optarg);
      break;

    case 'b':
      batch_size = atoi(optarg);
      break;

    case 'n':
      num_threads = atoi(optarg);
      break;
    
    case 'm':
      matrix_threads = atoi(optarg);
      break;

    case 'c':
      use_cuda = true;
      break;

    default:
      break;
    }
  }

  if (use_cuda) { // GPU inference

    LSTM<CudaMatrix> lstm(hidden_size, hidden_size, depth);

    std::vector<CudaMatrix> inputs;
    for (int t = 0; t < timesteps; t++) {
      CudaMatrix input(hidden_size, batch_size);
      input._init_random();
      inputs.push_back(input);
    }

    CudaMatrix output(hidden_size, batch_size);
    CudaMatrix gt_output(hidden_size, batch_size);

    auto start_time = Clock::now();

    if (do_par == 0) {
      lstm.forward(inputs, output);
    } else if (do_par == 1) {
      lstm.forward_par1(inputs, output, num_threads);
    } else {
      lstm.forward_par2(inputs, output, num_threads);
    }

    auto end_time = Clock::now();

    lstm.forward(inputs, gt_output);
    bool correct = output == gt_output;

    // DEBUG
    // double *data = output.get_data();
    // double *gt_data = gt_output.get_data();
    // arrayPrint(data, hidden_size * batch_size);
    // std::cout << " ----- " << std::endl;
    // arrayPrint(gt_data, hidden_size * batch_size);

    std::cout << "Correctness: " << correct << std::endl;
    auto total_time = duration_cast<dsec>(end_time - start_time).count();
    std::cout << "Total time: " << total_time << std::endl;

  } else { // CPU inference

    // Set OpenMP threads
    omp_set_num_threads(matrix_threads);
    std::cout << "Number of matmul threads: " << Eigen::nbThreads() << std::endl;

    LSTM<EigenMatrix> lstm(hidden_size, hidden_size, depth);

    std::vector<EigenMatrix> inputs;
    for (int t = 0; t < timesteps; t++) {
      EigenMatrix input(hidden_size, batch_size);
      input._init_random();
      inputs.push_back(input);
    }

    EigenMatrix output(hidden_size, batch_size);
    EigenMatrix gt_output(hidden_size, batch_size);

    auto start_time = Clock::now();

    if (do_par == 0) {
      lstm.forward(inputs, output);
    } else if (do_par == 1) {
      lstm.forward_par1(inputs, output, num_threads);
    } else {
      lstm.forward_par2(inputs, output, num_threads);
    }

    auto end_time = Clock::now();

    if (check_correct) {
      lstm.forward(inputs, gt_output);
      bool correct = output == gt_output;
      std::cout << "Correctness: " << correct << std::endl;
    }

    // DEBUG
    // double *data = output.get_data();
    // double *gt_data = gt_output.get_data();
    // arrayPrint(data, hidden_size * batch_size);
    // std::cout << " ----- " << std::endl;
    // arrayPrint(gt_data, hidden_size * batch_size);

    auto total_time = duration_cast<dsec>(end_time - start_time).count();
    std::cout << "Total time: " << total_time << std::endl;

  }

  return 0;
}
