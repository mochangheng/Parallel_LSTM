cmake_minimum_required(VERSION 3.10)

# set the project name
project(parallel_lstm)
enable_language(CUDA)

set(SOURCES
    src/cuda_kernels.cu
    src/eigen_matrix.cpp
    src/cuda_matrix.cpp
    src/lstm.cpp
    src/lstm_par.cpp
    src/lstm_par2.cpp
    src/main.cpp
    src/nnlib.cpp
)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# add the executable
add_executable(parallel_lstm ${SOURCES})

target_include_directories(parallel_lstm
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(parallel_lstm PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(parallel_lstm PRIVATE CUDA::cudart CUDA::cublas)
