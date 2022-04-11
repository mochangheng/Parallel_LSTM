cmake_minimum_required(VERSION 3.10)

# set the project name
project(parallel_lstm)

set(SOURCES
    src/eigen_matrix.cpp
    src/lstm.cpp
    src/lstm_par.cpp
    src/main.cpp
    src/nnlib.cpp
)

# add the executable
add_executable(parallel_lstm ${SOURCES})

target_include_directories(parallel_lstm
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)
