cmake_minimum_required(VERSION 3.16)
project(sdf_gradients LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CPU tuning — applied globally so sdf_core and sdf_cpp share the same
# Eigen vectorization. Per-target -march on sdf_core only causes Eigen
# ABI mismatch across TU boundaries → segfault in PUInterpolator::fit.
if(NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-march=native)
endif()

# ── Options ──────────────────────────────────────────────────────────
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python module" ON)
option(USE_OPENMP            "Enable OpenMP parallelism"    ON)

# ── FetchContent ─────────────────────────────────────────────────────
include(FetchContent)

# ── Eigen3 ───────────────────────────────────────────────────────────
# Prefer the fork's pinned Eigen (foreign 3.4.90 dev build caused CGAL
# MES memory blowup ~100GB). Fall back to system, then fetch 3.4.0.
set(FORK_EIGEN "${CMAKE_CURRENT_SOURCE_DIR}/../../sdf/maximal-empty-spheres/cgal/build/_deps/eigen-src")
if(EXISTS "${FORK_EIGEN}/Eigen/Core")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FORK_EIGEN}")
    message(STATUS "Using fork Eigen at ${FORK_EIGEN}")
else()
    find_package(Eigen3 3.3 QUIET NO_MODULE)
    if(NOT Eigen3_FOUND)
        message(STATUS "Eigen3 not found — fetching 3.4.0")
        set(EIGEN_BUILD_DOC    OFF CACHE BOOL "" FORCE)
        set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING      OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG        3.4.0
            GIT_SHALLOW    TRUE)
        FetchContent_MakeAvailable(eigen)
    endif()
endif()

# ── CGAL ─────────────────────────────────────────────────────────────
# Header-only for our usage. GMP + MPFR are compiled libs — they must
# come from the system (apt: libgmp-dev libmpfr-dev / brew: gmp mpfr).
# The MES fork provides the unreleased Maximal_empty_spheres package.
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
    HINTS /opt/homebrew/include /usr/local/include /usr/include)
find_library(GMP_LIBRARY   NAMES gmp
    HINTS /opt/homebrew/lib /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
find_library(GMPXX_LIBRARY NAMES gmpxx
    HINTS /opt/homebrew/lib /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
find_library(MPFR_LIBRARY  NAMES mpfr
    HINTS /opt/homebrew/lib /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)

if(NOT CGAL_INCLUDE_DIR)
    message(STATUS "CGAL headers not found — fetching v5.6.2 (large download)")
    FetchContent_Declare(cgal
        GIT_REPOSITORY https://github.com/CGAL/cgal.git
        GIT_TAG        v5.6.2
        GIT_SHALLOW    TRUE)
    FetchContent_Populate(cgal)
    # CGAL git repo uses per-package layout: Package/include/CGAL/...
    file(GLOB _cgal_pkg_incs LIST_DIRECTORIES true "${cgal_SOURCE_DIR}/*/include")
    find_path(CGAL_INCLUDE_DIR CGAL/Epick_d.h
        HINTS ${_cgal_pkg_incs} NO_DEFAULT_PATH)
endif()

if(CGAL_INCLUDE_DIR AND GMP_LIBRARY AND MPFR_LIBRARY)
    set(HAVE_MES_CONTACT TRUE)
    message(STATUS "MES contact enabled (CGAL at ${CGAL_INCLUDE_DIR})")
else()
    set(HAVE_MES_CONTACT FALSE)
    if(NOT CGAL_INCLUDE_DIR)
        message(STATUS "MES contact disabled — CGAL headers missing")
    else()
        message(STATUS "MES contact disabled — GMP/MPFR missing (apt: libgmp-dev libmpfr-dev)")
    endif()
endif()

# ── nanoflann ────────────────────────────────────────────────────────
find_path(NANOFLANN_INCLUDE_DIR nanoflann.hpp
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanoflann/include
    NO_DEFAULT_PATH)
if(NOT NANOFLANN_INCLUDE_DIR)
    message(STATUS "nanoflann not found — fetching v1.5.5")
    set(NANOFLANN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NANOFLANN_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(nanoflann
        GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
        GIT_TAG        v1.5.5
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(nanoflann)
    set(NANOFLANN_INCLUDE_DIR ${nanoflann_SOURCE_DIR}/include)
endif()

# ── libigl ───────────────────────────────────────────────────────────
FetchContent_Declare(libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG        v2.5.0
    GIT_SHALLOW    TRUE)
FetchContent_GetProperties(libigl)
if(NOT libigl_POPULATED)
    FetchContent_Populate(libigl)
endif()
set(LIBIGL_INCLUDE_DIR ${libigl_SOURCE_DIR}/include)

# ── OpenMP ───────────────────────────────────────────────────────────
if(USE_OPENMP)
    find_package(OpenMP)
    if(NOT OpenMP_CXX_FOUND AND APPLE)
        execute_process(COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
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

# ── pybind11 ─────────────────────────────────────────────────────────
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        execute_process(
            COMMAND python3 -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _pybind11_pip_rc)
        if(_pybind11_pip_rc EQUAL 0 AND EXISTS "${_pybind11_dir}")
            set(pybind11_DIR "${_pybind11_dir}")
            find_package(pybind11 CONFIG QUIET)
        endif()
    endif()
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found — fetching v2.13.6")
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.13.6
            GIT_SHALLOW    TRUE)
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# ── Source dirs ──────────────────────────────────────────────────────
set(BENCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../bench)
set(ROOT_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/..)

# ── Core static library ──────────────────────────────────────────────
add_library(sdf_core STATIC
    src/interpolator.cpp
    src/duchon_interpolator.cpp
    src/pu_interpolator.cpp
    src/visibility.cpp
    src/clamp.cpp
    src/optimization.cpp
    src/main_algorithm.cpp
    src/sphere_intersect_core.cpp
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

# CPU-specific tuning. Must apply to ALL targets (sdf_core AND sdf_cpp) —
# inconsistent vectorization between TUs causes Eigen ABI mismatch
# (segfault in PUInterpolator::fit when types cross TU boundaries).
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(sdf_core PRIVATE -mcpu=apple-m1)
endif()

if(HAVE_MES_CONTACT)
    # Fork's Maximal_empty_spheres headers MUST precede system CGAL so
    # the unreleased package is picked up first.
    target_include_directories(sdf_core SYSTEM PUBLIC
        ${MES_FORK_INCLUDES} ${CGAL_INCLUDE_DIR})
    target_compile_definitions(sdf_core PUBLIC HAVE_MES_CONTACT CGAL_USE_GMPXX=1)
    target_link_libraries(sdf_core PUBLIC
        ${GMPXX_LIBRARY} ${MPFR_LIBRARY} ${GMP_LIBRARY})
    # EIGEN_USE_BLAS (Accelerate) interacts pathologically with CGAL MES,
    # blowing up memory. Undefine it for this TU only.
    set_source_files_properties(src/mes_contact_core.cpp PROPERTIES
        COMPILE_OPTIONS "-UEIGEN_USE_BLAS")
endif()

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(sdf_core PUBLIC ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(sdf_core PUBLIC EIGEN_USE_BLAS)
    endif()
elseif(UNIX)
    # Linux equivalent of Mac's Accelerate. Look for OpenBLAS in:
    #   1. Bundled PyPI package at third_party/openblas (sudo-free option:
    #      `uv pip install --target ... scipy_openblas32`)
    #   2. System install (apt: libopenblas-dev)
    set(BUNDLED_OPENBLAS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openblas/site-packages/scipy_openblas32")
    if(EXISTS "${BUNDLED_OPENBLAS}/lib/libscipy_openblas.so")
        set(OPENBLAS_LIBRARY "${BUNDLED_OPENBLAS}/lib/libscipy_openblas.so")
        set(OPENBLAS_INCLUDE_DIR "${BUNDLED_OPENBLAS}/include")
        message(STATUS "Using bundled scipy_openblas32 at ${BUNDLED_OPENBLAS}")
    else()
        find_library(OPENBLAS_LIBRARY NAMES openblas
            HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
        find_path(OPENBLAS_INCLUDE_DIR cblas.h
            HINTS /usr/include /usr/include/x86_64-linux-gnu /usr/local/include)
    endif()
    if(OPENBLAS_LIBRARY)
        # OpenBLAS is detected but NOT auto-linked. The hot path here is PU
        # patches of 100–700 rows; Eigen's internal vectorized kernels
        # (AVX2/AVX-512 via -march=native) beat the dispatch overhead of a
        # BLAS call at that size. Auto-linking it measurably regressed
        # runtime in our benchmarks. Future code that does need real BLAS
        # (large dense solves) can opt in per-target.
        message(STATUS "OpenBLAS detected at ${OPENBLAS_LIBRARY} — not linked (use Eigen-internal kernels)")
    endif()
endif()

if(OpenMP_CXX_FOUND)
    if(APPLE AND NOT TARGET OpenMP::OpenMP_CXX)
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
