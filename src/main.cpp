#include <iostream>
#include <Eigen/Dense>
#include <eigen_matrix.hpp>
#include <lstm.hpp>
#include <cmath>
#include <vector>
#include <chrono>

using Eigen::MatrixXd;
using namespace std::chrono;
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> dsec;


int main() {
  const int depth = 4;
  const int hidden_size = 128;
  const int timesteps = 128;
  const bool do_par = true;

  LSTM<EigenMatrix> lstm(hidden_size, hidden_size, depth);

  std::vector<EigenMatrix> inputs;
  for (int t = 0; t < timesteps; t++) {
    EigenMatrix input(hidden_size, 1);
    input._init_random();
    inputs.push_back(input);
  }

  EigenMatrix output(hidden_size, 1);

  auto start_time = Clock::now();

  if (do_par) {
    lstm.forward_par1(inputs, output);
  } else {
    lstm.forward(inputs, output);
  }

  auto end_time = Clock::now();
  auto total_time = duration_cast<dsec>(end_time - start_time).count();
  std::cout << "Total time: " << total_time << std::endl;

  return 0;
}
