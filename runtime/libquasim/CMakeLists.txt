cmake_minimum_required(VERSION 3.20)
project(libquasim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(quasim STATIC
    src/quasim.cpp
    src/scheduler.cpp
    src/telemetry.cpp
)

target_include_directories(quasim PUBLIC include)

target_compile_definitions(quasim PRIVATE QUASIM_VERSION="1.0.0")

add_executable(quasim_cli src/main.cpp)

target_link_libraries(quasim_cli PRIVATE quasim)

include(CTest)
add_executable(quasim_tests tests/quasim_tests.cpp)
target_link_libraries(quasim_tests PRIVATE quasim)
add_test(NAME quasim_unit COMMAND quasim_tests)
