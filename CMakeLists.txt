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
find_package(CUDA REQUIRED)

# add the executable
add_executable(parallel_lstm ${SOURCES})

include_directories("${CUDA_INCLUDE_DIRS}")
target_include_directories(parallel_lstm
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(parallel_lstm PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(parallel_lstm PRIVATE cudart cublas)
