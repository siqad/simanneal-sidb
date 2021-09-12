cmake_minimum_required(VERSION 3.13)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12")
    cmake_policy(SET CMP0074 NEW)
endif()

# TODO compile CUDA and C++ files separately and link together
# https://gist.github.com/erikzenker/713c4ff76949058d5d5d
#SET(CUDA_SEPARABLE_COMPILATION ON)

# for release builds, add flag "-DCMAKE_BUILD_TYPE=Release" (without quotes) when running cmake.

project(simanneal LANGUAGES CXX CUDA)
# TODO add way to push version number into source code
#list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50 -rdc=true)
#set(CUDA_SEPARABLE_COMPILATION ON)

# CUDA settings
#string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_50,code=sm_50 -rdc=true")
#set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
#target_compile_options(simanneal PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                       --generate-line-info
#                       --use_fast_math
#                       --relocatable-device-code=true
#                       >)

##find_package(CUDA QUIET REQUIRED)
##set(CUDA_SEPARABLE_COMPILATION ON)
##find_package(CUDA QUIET REQUIRED)
##set(CUDA_PROPAGATE_HOST_FLAGS OFF)
##string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_50,code=sm_50 -rdc=true")

# Other dependencies
if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
endif()
find_package(Boost REQUIRED)

include_directories(
    .
    src
    swig
    ${Boost_INCLUDE_DIR}
)

# shared source files
set(SOURCES
    src/main.cc
    src/interface.cc 
    src/simanneal_cuda.cu
    src/global.cc
    src/siqadconn.cc
    #src/simanneal_cuda.cu
    #src/global.cc
)

if(SKBUILD)
    # building Python wrapper

    #add_definitions(-Wall -g -std=c++11)
    
    find_package(SWIG REQUIRED)
    include(${SWIG_USE_FILE})

    find_package(PythonLibs REQUIRED)
    include_directories(${PYTHON_INCLUDE_PATH})

    set_property(SOURCE swig/simanneal.i PROPERTY CPLUSPLUS ON)
    file(COPY swig/__init__.py DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/swig")

    swig_add_library(simanneal LANGUAGE python SOURCES swig/simanneal.i ${SOURCES})

    swig_link_libraries(simanneal ${Boost_LIBRARIES})

    install(TARGETS ${SWIG_MODULE_simanneal_REAL_NAME}
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../cmake-install/pysimanneal)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/simanneal.py
                  ${CMAKE_CURRENT_BINARY_DIR}/swig/__init__.py
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../cmake-install/pysimanneal)

else()
    # building binary

    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)

    add_executable(simanneal ${SOURCES})

    set_target_properties(simanneal PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    #target_compile_options(${PROJECT_NAME} PRIVATE "SHELL:-gencode arch=compute_50,code=sm_50")
    # https://stackoverflow.com/questions/54504253/how-to-add-more-than-one-cuda-gencode-using-modern-cmake-per-target
    #target_compile_options(simanneal PRIVATE
    #    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_50,code=sm_50>"
    #)
    #target_compile_options(simanneal PUBLIC
    #    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_52,code=sm_52 -gencode arch=compute_50,code=sm_50>"
    #)
    #target_compile_options(simanneal PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_30,code=sm_30>)
    #SET(CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(simanneal
        PUBLIC
            # Maxwell Gen1
            #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_50,code=sm_50 --relocatable-device-code=true>
            # Maxwell Gen2
            #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_52,code=sm_52 --relocatable-device-code=true>
            # Pascal P100
            #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_60,code=sm_60 --relocatable-device-code=true>
            # Pascal
            #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=sm_61 --relocatable-device-code=true>
            # Volta
            #$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_70,code=compute_70 --relocatable-device-code=true>
            # 
            $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=compute_75>
    )

    #target_compile_options(${PROJECT_NAME} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_50,code=sm_50>)
    #target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    #    --generate-line-info
    #    --use_fast_math
    #    --relocatable-device-code=true
    #    >)
    #target_link_options(${PROJECT_NAME} PRIVATE "-arch=sm_50")
    #add_link_options("$<DEVICE_LINK:-arch=sm_50>")
    #add_link_options("-arch=sm_50>")
    #target_link_options(simanneal PRIVATE $<$<LINK_LANGUAGE:CUDA>:-arch sm_50>)
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

endif()
