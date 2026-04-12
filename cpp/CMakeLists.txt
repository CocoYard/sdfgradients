cmake_minimum_required(VERSION 3.16)
project(sdf_gradients LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Options ──────────────────────────────────────────────────────────
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python module" ON)
option(USE_OPENMP            "Enable OpenMP parallelism"    ON)

# ── Dependencies ─────────────────────────────────────────────────────

# Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# nanoflann (header-only KDTree)
# Put nanoflann in third_party/ via:
#   git submodule add https://github.com/jlblancoc/nanoflann.git third_party/nanoflann
# Or just download the single header nanoflann.hpp into third_party/nanoflann/include/
set(NANOFLANN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanoflann/include)

# libigl (header-only; we only need igl/marching_cubes.h).
# Fetched via CMake FetchContent — no build steps, just the include tree.
include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG        v2.5.0
)
FetchContent_GetProperties(libigl)
if(NOT libigl_POPULATED)
    FetchContent_Populate(libigl)
endif()
set(LIBIGL_INCLUDE_DIR ${libigl_SOURCE_DIR}/include)

# OpenMP
if(USE_OPENMP)
    find_package(OpenMP)
    if(NOT OpenMP_CXX_FOUND)
        # macOS with Homebrew libomp
        if(APPLE)
            execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(EXISTS "${LIBOMP_PREFIX}")
                set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
                set(OpenMP_CXX_LIB_NAMES "omp")
                set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
                include_directories("${LIBOMP_PREFIX}/include")
                link_directories("${LIBOMP_PREFIX}/lib")
                set(OpenMP_CXX_FOUND TRUE)
            endif()
        endif()
    endif()
endif()

# pybind11
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        # Fallback: use pybind11 installed via pip
        execute_process(
            COMMAND python3 -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        find_package(pybind11 CONFIG REQUIRED)
    endif()
endif()

# ── Existing C++ modules (reuse from parent project) ─────────────────
# These are compiled as part of our library, stripping the PYBIND11_MODULE
# and exposing pure C++ functions. We include their source directly.
set(BENCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../bench)
set(ROOT_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/..)

# ── Core static library ─────────────────────────────────────────────
add_library(sdf_core STATIC
    src/interpolator.cpp
    src/duchon_interpolator.cpp
    src/pu_interpolator.cpp
    src/visibility.cpp
    src/clamp.cpp
    src/optimization.cpp
    src/main_algorithm.cpp
    src/existing_modules_adapter.cpp
)

target_include_directories(sdf_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${NANOFLANN_INCLUDE_DIR}
    ${LIBIGL_INCLUDE_DIR}
    ${BENCH_DIR}
    ${ROOT_DIR}
)

target_link_libraries(sdf_core PUBLIC Eigen3::Eigen)

# Use Apple Accelerate for BLAS/LAPACK (matches numpy's backend)
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(sdf_core PUBLIC ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(sdf_core PUBLIC EIGEN_USE_BLAS)
    endif()
endif()

if(OpenMP_CXX_FOUND)
    if(APPLE AND NOT TARGET OpenMP::OpenMP_CXX)
        # Homebrew libomp: flags must be separate tokens
        target_compile_options(sdf_core PUBLIC -Xpreprocessor -fopenmp)
        target_include_directories(sdf_core PUBLIC "${LIBOMP_PREFIX}/include")
        target_link_libraries(sdf_core PUBLIC "${LIBOMP_PREFIX}/lib/libomp.dylib")
    else()
        target_link_libraries(sdf_core PUBLIC OpenMP::OpenMP_CXX)
    endif()
    target_compile_definitions(sdf_core PUBLIC USE_OPENMP)
endif()

# ── Python bindings ──────────────────────────────────────────────────
if(BUILD_PYTHON_BINDINGS)
    pybind11_add_module(sdf_cpp bindings/pybind_main.cpp)
    target_link_libraries(sdf_cpp PRIVATE sdf_core)
endif()
