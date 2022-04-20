#include <iostream>
#include <Eigen/Dense>
#include <eigen_matrix.hpp>
#include <lstm.hpp>
#include <cmath>
#include <vector>
#include <chrono>
#include <getopt.h>
#include <omp.h>

using Eigen::MatrixXd;
using namespace std::chrono;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> dsec;

int main(int argc, char* argv[]) {
  bool do_par = true;
  int depth = 4;
  int hidden_size = 128;
  int timesteps = 128;
  int batch_size = 1;

  int num_threads = 1;
  int matrix_threads = 1;
  int c;

  while ((c = getopt(argc, argv, "d:h:t:b:n:m:")) != -1) {
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

    default:
      break;
    }
  }

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

  auto start_time = Clock::now();

  if (do_par) {
    lstm.forward_par1(inputs, output, num_threads);
  }
  else {
    lstm.forward(inputs, output);
  }

  auto end_time = Clock::now();
  auto total_time = duration_cast<dsec>(end_time - start_time).count();
  std::cout << "Total time: " << total_time << std::endl;

  return 0;
}
