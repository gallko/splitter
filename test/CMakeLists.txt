project(TestSplitter)

file(GLOB TEST_SOURCES
        ./main.cpp
        ./src/*.cpp)

include_directories(../include)

add_executable(${PROJECT_NAME} ${TEST_SOURCES})

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PUBLIC ${BINARY_NAME} gtest)