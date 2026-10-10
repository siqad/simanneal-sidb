cmake_minimum_required(VERSION 3.11)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12")
    cmake_policy(SET CMP0074 NEW)
endif()
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.27")
    cmake_policy(SET CMP0144 NEW)
endif()

# for release builds, add flag "-DCMAKE_BUILD_TYPE=Release" (without quotes) when running cmake.

project(simanneal CXX)

# Optional whole-program optimization. Runtime kernels retain strict FP flags.
option(SIMANNEAL_ENABLE_LTO "Enable supported link-time optimization" OFF)
if(SIMANNEAL_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT SIMANNEAL_LTO_SUPPORTED
                        OUTPUT SIMANNEAL_LTO_ERROR LANGUAGES CXX)
    if(NOT SIMANNEAL_LTO_SUPPORTED)
        message(FATAL_ERROR "Link-time optimization requested but unsupported: ${SIMANNEAL_LTO_ERROR}")
    endif()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_VERSION VERSION_LESS "3.13")
            message(FATAL_ERROR "GNU link-time optimization requires CMake >= 3.13 for bounded link parallelism")
        endif()
        # Match qualification; bound GCC's LTO partitions independently of -j.
        add_compile_options(-flto=4)
        add_link_options(-flto=4)
    endif()
endif()
# TODO add way to push version number into source code

if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
endif()
find_package(Boost COMPONENTS random REQUIRED)
find_package(Threads REQUIRED)

# Explicitly enabled supported OpenBLAS installation; never inspect Python wheels.
option(SIMANNEAL_ENABLE_OPENBLAS "Enable explicit OpenBLAS population backends" OFF)
option(SIMANNEAL_NATIVE_ARCH "Compile for the build host (not portable binaries)" OFF)
if(SIMANNEAL_ENABLE_OPENBLAS)
    find_path(SIMANNEAL_OPENBLAS_INCLUDE_DIR cblas.h
              HINTS ${SIMANNEAL_OPENBLAS_ROOT}/include
              PATH_SUFFIXES openblas)
    find_library(SIMANNEAL_OPENBLAS_LIBRARY NAMES openblas
                 HINTS ${SIMANNEAL_OPENBLAS_ROOT}/lib ${SIMANNEAL_OPENBLAS_ROOT}/lib64)
    if(NOT SIMANNEAL_OPENBLAS_INCLUDE_DIR OR NOT SIMANNEAL_OPENBLAS_LIBRARY)
        message(FATAL_ERROR "OpenBLAS requested: provide its supported cblas.h and library via SIMANNEAL_OPENBLAS_ROOT or SIMANNEAL_OPENBLAS_INCLUDE_DIR/LIBRARY")
    endif()
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES ${SIMANNEAL_OPENBLAS_INCLUDE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES ${SIMANNEAL_OPENBLAS_LIBRARY} ${CMAKE_THREAD_LIBS_INIT})
    # Recheck when users change installations or ABI between configurations.
    unset(SIMANNEAL_OPENBLAS_CBLAS_WORKS CACHE)
    check_cxx_source_compiles("#include <cblas.h>
int main() {
  double a[1]={1}, x[1]={1}, y[1]={0};
  openblas_set_num_threads(1);
  cblas_dgemv(CblasRowMajor,CblasNoTrans,1,1,1,a,1,x,1,0,y,1);
  cblas_dsymv(CblasRowMajor,CblasUpper,1,1,a,1,x,1,0,y,1);
  return openblas_get_num_threads()!=1;
}" SIMANNEAL_OPENBLAS_CBLAS_WORKS)
    unset(CMAKE_REQUIRED_INCLUDES)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(NOT SIMANNEAL_OPENBLAS_CBLAS_WORKS)
        message(FATAL_ERROR "OpenBLAS header/library must provide DGEMV, DSYMV and supported thread-control APIs")
    endif()
endif()

# The portable implementation is always available. Apple builds may opt out.
option(SIMANNEAL_ENABLE_ACCELERATE "Enable the Apple Accelerate population backend" ${APPLE})
if(APPLE AND SIMANNEAL_ENABLE_ACCELERATE AND SIMANNEAL_ENABLE_OPENBLAS)
    message(FATAL_ERROR "Accelerate and OpenBLAS export identical CBLAS symbols. Set SIMANNEAL_ENABLE_ACCELERATE=OFF for an OpenBLAS build.")
endif()
if(APPLE AND SIMANNEAL_ENABLE_ACCELERATE)
    find_library(SIMANNEAL_ACCELERATE_FRAMEWORK Accelerate)
    if(NOT SIMANNEAL_ACCELERATE_FRAMEWORK)
        message(FATAL_ERROR "SIMANNEAL_ENABLE_ACCELERATE requires the Accelerate framework")
    endif()
endif()
function(simanneal_configure_population_backend target)
    target_sources(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/population_blas.cc)
    target_link_libraries(${target} Threads::Threads)
    if(MSVC)
        target_compile_options(${target} PRIVATE /fp:strict)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -fno-fast-math -ffp-contract=off)
        if(SIMANNEAL_NATIVE_ARCH)
            target_compile_options(${target} PRIVATE -march=native)
        endif()
    elseif(SIMANNEAL_NATIVE_ARCH)
        message(FATAL_ERROR "SIMANNEAL_NATIVE_ARCH is supported only by GNU/Clang compilers")
    endif()
    if(MSVC AND SIMANNEAL_NATIVE_ARCH)
        message(FATAL_ERROR "SIMANNEAL_NATIVE_ARCH requires GNU/Clang; choose an explicit MSVC architecture separately")
    endif()
    if(SIMANNEAL_ENABLE_OPENBLAS)
        target_compile_definitions(${target} PUBLIC SIMANNEAL_HAVE_OPENBLAS=1)
        target_include_directories(${target} PRIVATE ${SIMANNEAL_OPENBLAS_INCLUDE_DIR})
        target_link_libraries(${target} ${SIMANNEAL_OPENBLAS_LIBRARY})
    endif()
    if(APPLE AND SIMANNEAL_ENABLE_ACCELERATE)
        target_compile_definitions(${target} PUBLIC SIMANNEAL_HAVE_ACCELERATE=1)
        target_link_libraries(${target} ${SIMANNEAL_ACCELERATE_FRAMEWORK})
    endif()
endfunction()

#add_definitions( -DAPP_VERSION=\"0.1.0\" -DAPPLICATION_NAME=\"SiQAD\" -DORGANIZATION_NAME=\"WalusLab\" )

include_directories(
    .
    src
    swig
    ${Boost_INCLUDE_DIR}
)

if(SKBUILD)
    add_definitions(-Wall -g -std=c++11)

    # set_target_properties(libraryname PROPERTIES LINK_FLAGS "-Wl,-znodelete")
    
    find_package(SWIG REQUIRED)
    include(${SWIG_USE_FILE})

    find_package(PythonLibs REQUIRED)
    include_directories(${PYTHON_INCLUDE_PATH})

    set_property(SOURCE swig/simanneal.i PROPERTY CPLUSPLUS ON)
    file(COPY swig/__init__.py DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/swig")

    swig_add_library(simanneal LANGUAGE python SOURCES swig/simanneal.i src/simanneal.cc src/global.cc src/physical_repair.cc src/refinement.cc)
    simanneal_configure_population_backend(${SWIG_MODULE_simanneal_REAL_NAME})

    if (WIN32)
        swig_link_libraries(simanneal ${Boost_LIBRARIES} ${PYTHON_LIBRARY})
    else()
        swig_link_libraries(simanneal ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
        if(APPLE)
            # Resolve Python symbols from the importing interpreter. Linking a
            # second libpython crashes statically embedded Python distributions.
            set_property(TARGET ${SWIG_MODULE_simanneal_REAL_NAME}
                         APPEND_STRING PROPERTY LINK_FLAGS " -undefined dynamic_lookup")
        endif()
    endif()

    # set_target_properties(simanneal PROPERTIES LINK_FLAGS "-lpython,-dynamiclib")

    install(TARGETS ${SWIG_MODULE_simanneal_REAL_NAME}
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../cmake-install/pysimanneal)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/simanneal.py
                  ${CMAKE_CURRENT_BINARY_DIR}/swig/__init__.py
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../cmake-install/pysimanneal)

else()

    if(DEFINED SIQAD_PLUGINS_ROOT)
        set(SIMANNEAL_INSTALL_DIR "${SIQAD_PLUGINS_ROOT}/simanneal")
    elseif(CMAKE_BUILD_TYPE MATCHES Debug)
        set(SIMANNEAL_INSTALL_DIR "debug")
    else()
        set(SIMANNEAL_INSTALL_DIR "release")
    ENDIF()

    add_subdirectory(libs/siqadconn)

    set(SIMANNEAL_SOURCES
        src/main.cc
        src/global.cc
        src/interface.cc
        src/simanneal.cc
        src/physical_repair.cc
        src/refinement.cc
    )

    set(SIMANNEAL_TESTS
        tests/simanneal.test.cc
        tests/population_backend.test.cc
        tests/population_blas.test.cc
        tests/refinement.test.cc
        tests/validated_search.test.cc
        tests/result_export.test.cc
        tests/population_probability.test.cc
        tests/parameter_storage.test.cc
        tests/kernel_defaults.test.cc
        tests/auto_budget.test.cc

        # HEADERS
        libs/catch2/catch.hpp
    )

    add_library(simanneal_lib STATIC ${SIMANNEAL_SOURCES})
    target_link_libraries(simanneal_lib siqadconn ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    simanneal_configure_population_backend(simanneal_lib)

    add_executable(simanneal src/main.cc)
    target_link_libraries(simanneal simanneal_lib)

    add_dependencies(simanneal siqadconn)

    set_property(TARGET simanneal_lib PROPERTY CXX_STANDARD 11)
    set_property(TARGET simanneal_lib PROPERTY CXX_STANDARD_REQUIRED True)
    set_property(TARGET simanneal PROPERTY CXX_STANDARD 11)
    set_property(TARGET simanneal PROPERTY CXX_STANDARD_REQUIRED True)

    # copy SiQAD integration files over to compiled binary dir
    add_custom_command(
        TARGET simanneal POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/simanneal.physeng
                ${CMAKE_CURRENT_SOURCE_DIR}/ubc_logo.png
                ${CMAKE_CURRENT_BINARY_DIR})

    option(BUILD_TEST "Build the test program." ON)
    if (BUILD_TEST)
        add_executable(simanneal_tests tests/tests_main.cc ${SIMANNEAL_TESTS})
        target_link_libraries(simanneal_tests simanneal_lib)
        target_compile_definitions(simanneal_tests PRIVATE
            SIMANNEAL_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

        add_dependencies(simanneal_tests simanneal)

        set_property(TARGET simanneal_tests PROPERTY CXX_STANDARD 11)
        set_property(TARGET simanneal_tests PROPERTY CXX_STANDARD_REQUIRED True)

        enable_testing()
        add_test(simanneal_tests simanneal_tests)
        add_executable(simanneal_affinity_tests tests/affinity_workers.test.cc)
        target_link_libraries(simanneal_affinity_tests Threads::Threads)
        set_property(TARGET simanneal_affinity_tests PROPERTY CXX_STANDARD 11)
        set_property(TARGET simanneal_affinity_tests PROPERTY CXX_STANDARD_REQUIRED True)
        add_test(simanneal_affinity_tests simanneal_affinity_tests)
        if(CMAKE_VERSION VERSION_LESS "3.12")
            find_package(PythonInterp 3 REQUIRED)
            set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
        else()
            find_package(Python3 COMPONENTS Interpreter REQUIRED)
        endif()
        add_test(NAME simanneal_cli_tests
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_integration.py
                    $<TARGET_FILE:simanneal> ${CMAKE_CURRENT_SOURCE_DIR})
        # add_custom_command(
        #     TARGET simanneal_tests
        #     POST_BUILD
        #     COMMAND ctest -C $<CONFIGURATION> --output-on-failure --verbose
        # )
    endif()

    install(TARGETS simanneal RUNTIME DESTINATION ${SIMANNEAL_INSTALL_DIR})
    install(FILES simanneal.physeng ubc_logo.png DESTINATION ${SIMANNEAL_INSTALL_DIR})

endif()

option(BUILD_BENCHMARK "Build the reproducible hopping benchmark driver." OFF)
if(BUILD_BENCHMARK AND NOT SKBUILD)
    add_executable(anneal_bench benchmarks/anneal_bench.cc)
    target_link_libraries(anneal_bench simanneal_lib)
    set_property(TARGET anneal_bench PROPERTY CXX_STANDARD 11)
endif()

option(BUILD_PRODUCTION_BENCHMARK "Build the complete-job search-profile benchmark." OFF)
if(BUILD_PRODUCTION_BENCHMARK AND NOT SKBUILD)
    add_executable(production_tts benchmarks/production_tts.cc)
    target_link_libraries(production_tts simanneal_lib)
    set_property(TARGET production_tts PROPERTY CXX_STANDARD 11)
    set_property(TARGET production_tts PROPERTY CXX_STANDARD_REQUIRED True)
    if(MSVC)
        target_compile_options(production_tts PRIVATE /fp:strict)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(production_tts PRIVATE -fno-fast-math -ffp-contract=off)
        if(SIMANNEAL_NATIVE_ARCH)
            target_compile_options(production_tts PRIVATE -march=native)
        endif()
    endif()
endif()
