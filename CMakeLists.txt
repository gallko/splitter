cmake_minimum_required(VERSION 3.10)
project(splitter)

set(CMAKE_CXX_STANDARD 17)

find_package(Threads)

include_directories(include)

file(GLOB ${PROJECT_NAME}_src
        ./*.cpp
        ./src/*.cpp)

add_executable(splitter ${${PROJECT_NAME}_src})
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} )