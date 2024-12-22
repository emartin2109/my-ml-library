cmake_minimum_required(VERSION 3.26)
project(lobotomyLearning CXX)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE SOURCES src/*.cpp)
add_executable(lobotomyLearning ${SOURCES})

target_compile_options(lobotomyLearning PRIVATE -W -Wall -Wextra -Wpedantic -Werror -g3) # TODO: remove -g3
target_include_directories(lobotomyLearning PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lobotomyLearning PRIVATE)