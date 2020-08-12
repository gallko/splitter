#pragma once

#define MEASURE_CALL(time, function, ...)  { \
    auto start_time = std::chrono::high_resolution_clock::now(); \
    function(__VA_ARGS__);                                     \
    auto end_time = std::chrono::high_resolution_clock::now();   \
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count(); }

#define MEASURE_CALL_R(time, result_function, function, ...)    { \
    auto start_time = std::chrono::high_resolution_clock::now(); \
    result_function = function(__VA_ARGS__);                   \
    auto end_time = std::chrono::high_resolution_clock::now();   \
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() ; }