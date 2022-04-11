# Layer-Parallel LSTMs

## Code Instruction

How to build and run the program

```bash
mkdir build
cd build/
cmake .. # only necessary to do once
cmake --build .
./parallel_lstm
```

## Summary

Long short-term memory (LSTM) is a neural network architecture used for sequence modeling. However, one notorious issue with LSTMs is that they are not easily parallelizable due to time dependencies. This seriously restricts their application because their training and inference is slow. In this project, we will attempt to parallelize LSTMs on different hardware by executing multiple layers in parallel.

## Background

LSTMs were created as an improvement over the vanilla recurrent neural network (RNN). It maintains a memory cell and adds an input gate, an output gate and a forget gate to interact with the memory. These improvements allows LSTMs to learn long-term relationships in sequence data and mitigates the vanishing/exploding gradient problems that RNNs are prone to.

More practically speaking, an LSTM cell takes in an input at a timestep, updates its internal state (memory cell) and gives an output. This is repeated for all subsequent time steps. Multiple layers of LSTM cells are stacked on top on each other to create a deep neural network that can extract complex relationships in time-series data.

## Challenge

LSTMs are difficult to parallelize due to time dependecies (i.e. executing time step t requires internal state from time step t-1). In fact, they are rarely parallelized over different time steps in practice. As a result, LSTMs are notoriously slow during both training and inference. However, multiple cells are often stacked on top of each other in an LSTM network. This presents an opportunity for parallelization since executing time step t only requires the internal state from the previous time step t-1 **at the same layer**. However, since we have to manage dependencies carefully, synchronization overhead may be signifcant, which we will have to reduce as much as possible.

## Resources

We plan on trying to parallelize over both CPU and GPU hardware to see which hardware type gives better speedup. We would start by using the GHC machines. LSTM models often do not have a ton of layers, so we would not likely not even use all of the cores. However, we would want to make sure our solution is scalable, so we would want to test this on a system with a large amount of cores, such as PSC.

We shouldn't need any data sets, as we would be testing on arbitrary data. We are not focused on training the model, just parallelizing the computations it performs.

## Goals and Deliverables

### 75%
Achieve some speedup by parallelizing over layers.

### 100%
Achieve good speedup relative to the amount of layers in the model (as long as there are enough cores). We will also investigate differences and similarities between the performance characteristics of using CPU vs. GPU.

### 125%
Everything in 100%, as well as exploring whether there are any other aspects of the model's computation (e.g. backpropagation) that allow for parallelization. In addition, we can investigate different methods of parallelization on single GPU vs. multi GPU machines.


## Schedule

(3/28) Implement the LSTM architecture

(4/4) Parallelize LSTM inference on CPU

(4/11) Optimize the performance of parallel LSTM on CPU

(4/18) Parallelize LSTM inference on GPU using CUDA and optimize for speedup

(4/25) Write report and prepare for presentation
