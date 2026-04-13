cmake_minimum_required(VERSION 3.16)
project(sdf_gradients LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Options ──────────────────────────────────────────────────────────
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python module" ON)
option(USE_OPENMP            "Enable OpenMP parallelism"    ON)

# ── Dependencies ─────────────────────────────────────────────────────

# Eigen3 — use the fork's bundled Eigen 3.4.1 if available. find_package
# picked up a foreign 3.4.90 dev build from an unrelated project, whose
# SVD/matrix-product memory behavior blew up CGAL MES by ~100GB.
set(FORK_EIGEN "${CMAKE_CURRENT_SOURCE_DIR}/../../sdf/maximal-empty-spheres/cgal/build/_deps/eigen-src")
if(EXISTS "${FORK_EIGEN}/Eigen/Core")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FORK_EIGEN}")
    message(STATUS "Using fork Eigen at ${FORK_EIGEN}")
else()
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()

# CGAL (header-only; needs GMP + MPFR at link time). Used by mes_contact_core.
# The Maximal_empty_spheres package is unreleased — we include it from the
# local fork at ../../sdf/maximal-empty-spheres/cgal. Core CGAL comes from brew.
set(MES_FORK_BASE "${CMAKE_CURRENT_SOURCE_DIR}/../../sdf/maximal-empty-spheres/cgal"
    CACHE PATH "Path to the maximal-empty-spheres CGAL fork")
set(MES_FORK_INCLUDES "")
if(EXISTS "${MES_FORK_BASE}/cgal")
    file(GLOB MES_MODULES RELATIVE "${MES_FORK_BASE}/cgal" "${MES_FORK_BASE}/cgal/*")
    foreach(m ${MES_MODULES})
        if(IS_DIRECTORY "${MES_FORK_BASE}/cgal/${m}/include")
            list(APPEND MES_FORK_INCLUDES "${MES_FORK_BASE}/cgal/${m}/include")
        endif()
    endforeach()
endif()
find_path(CGAL_INCLUDE_DIR CGAL/Epick_d.h
    HINTS /opt/homebrew/include /usr/local/include)
find_library(GMP_LIBRARY   NAMES gmp   HINTS /opt/homebrew/lib /usr/local/lib)
find_library(GMPXX_LIBRARY NAMES gmpxx HINTS /opt/homebrew/lib /usr/local/lib)
find_library(MPFR_LIBRARY  NAMES mpfr  HINTS /opt/homebrew/lib /usr/local/lib)
if(CGAL_INCLUDE_DIR AND GMP_LIBRARY AND MPFR_LIBRARY)
    set(HAVE_MES_CONTACT TRUE)
    message(STATUS "MES contact enabled (CGAL at ${CGAL_INCLUDE_DIR})")
else()
    set(HAVE_MES_CONTACT FALSE)
    message(STATUS "MES contact disabled (CGAL/GMP/MPFR not found)")
endif()

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
    src/mes_contact_core.cpp
)

target_include_directories(sdf_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${NANOFLANN_INCLUDE_DIR}
    ${LIBIGL_INCLUDE_DIR}
    ${BENCH_DIR}
    ${ROOT_DIR}
)

target_link_libraries(sdf_core PUBLIC Eigen3::Eigen)

if(HAVE_MES_CONTACT)
    # Fork's Maximal_empty_spheres headers MUST come before brew CGAL so the
    # unreleased package is picked up; brew CGAL provides the rest.
    target_include_directories(sdf_core SYSTEM PUBLIC
        ${MES_FORK_INCLUDES} ${CGAL_INCLUDE_DIR})
    target_compile_definitions(sdf_core PUBLIC HAVE_MES_CONTACT CGAL_USE_GMPXX=1)
    target_link_libraries(sdf_core PUBLIC
        ${GMPXX_LIBRARY} ${MPFR_LIBRARY} ${GMP_LIBRARY})
    # EIGEN_USE_BLAS (Accelerate) interacts pathologically with CGAL MES's
    # Eigen usage, blowing up memory. Undefine it for this TU only.
    set_source_files_properties(src/mes_contact_core.cpp PROPERTIES
        COMPILE_OPTIONS "-UEIGEN_USE_BLAS")
endif()

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
