file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.h *.cpp)
find_package(Threads)

set(SOURCES ${SOURCES})

include_directories(../include)

add_library(${BINARY_NAME} STATIC ${SOURCES})

target_compile_options(${BINARY_NAME} PRIVATE -Wall)
target_link_libraries(${BINARY_NAME} ${CMAKE_THREAD_LIBS_INIT} )
