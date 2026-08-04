cmake_minimum_required(VERSION 3.20)
project(
    mala
    LANGUAGES CXX
    VERSION   0.0.1
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(_dst "${CMAKE_SOURCE_DIR}/compile_commands.json")
    if(NOT EXISTS "${_dst}" OR NOT "${_dst}" STREQUAL "${CMAKE_BINARY_DIR}/compile_commands.json")
        file(CREATE_LINK
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${_dst}"
            SYMBOLIC
        )
    endif()
endif()

# Default to Release for single-config generators
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MALA_BUILD_TESTS "Build the test suite"                           ON)
option(MALA_BUILD_BENCH "Build the benchmarks"                           ON)
option(MALA_WERROR      "Treat compiler warnings as errors"              OFF)
option(MALA_NATIVE      "Optimize for the build machine (-march=native)" OFF)


# ––––––––––– Library –––––––––––––––––––––––––––

add_library(mala
    src/Cpu.cpp
    src/ThreadPool.cpp
)
add_library(mala::mala ALIAS mala)

target_include_directories(mala PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(mala PUBLIC cxx_std_23)

find_package(Threads REQUIRED)
target_link_libraries(mala PUBLIC Threads::Threads)

if(MSVC)
    target_compile_options(mala PRIVATE /W4)
else()
    target_compile_options(mala PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MALA_WERROR)
    if (MSVC)
        target_compile_options(mala PRIVATE /WX)
    else()
        target_compile_options(mala PRIVATE -Werror)
    endif()
endif()

if(MALA_NATIVE AND NOT MSVC)
    target_compile_options(mala PRIVATE -march=native)
endif()


# ––––––––––– Tests ––––––––––––––––––––––––––––––––––

if(MALA_BUILD_TESTS)
    enable_testing()
    add_executable(mala_tests
        tests/TestMain.cpp
        tests/TestPlatform.cpp
    )
    target_link_libraries(mala_tests PRIVATE mala::mala)
    target_include_directories(mala_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_test(NAME mala_tests COMMAND mala_tests)
endif()


# ––––––––––– Benchmarks –––––––––––––––––––––––––––––––

if(MALA_BUILD_BENCH)
    add_executable(mala_bench
        bench/BenchPlatform.cpp
    )
    target_link_libraries(mala_bench PRIVATE mala::mala)
    target_include_directories(mala_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/bench)
    if(NOT MSVC)
        target_compile_options(mala_bench PRIVATE -O3)
        if(MALA_NATIVE)
            target_compile_options(mala_bench PRIVATE -march=native)
        endif()
    endif()
endif()
