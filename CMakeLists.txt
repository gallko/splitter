cmake_minimum_required(VERSION 3.17)
project(splitter)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

file(GLOB ${PROJECT_NAME}_src
        ./*.cpp
        ./src/*.cpp)

add_executable(splitter ${${PROJECT_NAME}_src})