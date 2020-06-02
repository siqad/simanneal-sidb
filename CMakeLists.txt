cmake_minimum_required(VERSION 3.10)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12")
    cmake_policy(SET CMP0074 NEW)
endif()

# for release builds, add flag "-DCMAKE_BUILD_TYPE=Release" (without quotes) when running cmake.

project(simanneal CXX)
# TODO add way to push version number into source code

#find_package(Boost COMPONENTS random REQUIRED)
if (WIN32)
    set(Boost_USE_STATIC_LIBS ON)
endif()
find_package(Boost COMPONENTS random REQUIRED)
find_package(Threads REQUIRED)

#add_definitions( -DAPP_VERSION=\"0.1.0\" -DAPPLICATION_NAME=\"SiQAD\" -DORGANIZATION_NAME=\"WalusLab\" )

include_directories(
    .
    src
    swig
    ${Boost_INCLUDE_DIR}
)

if(SKBUILD)
    add_definitions(-Wall -g -std=c++11)
    
    find_package(SWIG REQUIRED)
    include(${SWIG_USE_FILE})

    find_package(PythonLibs REQUIRED)
    include_directories(${PYTHON_INCLUDE_PATH})

    set_property(SOURCE swig/simanneal.i PROPERTY CPLUSPLUS ON)
    file(COPY swig/__init__.py DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/swig")

    swig_add_library(simanneal LANGUAGE python SOURCES swig/simanneal.i src/simanneal.cc src/global.cc)

    swig_link_libraries(simanneal ${Boost_LIBRARIES})

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

    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    #set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    #set(CMAKE_CXX_FLAGS_DEBUG "-g")
    #set(CMAKE_CXX_FLAGS_RELEASE "-O3")

    set(SOURCES
        src/main.cc
        src/global.cc
        src/interface.cc
        src/simanneal.cc
        src/siqadconn.cc
    )


    add_executable(simanneal ${SOURCES})

    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

    add_custom_command(
        TARGET simanneal POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/simanneal.physeng
                ${CMAKE_CURRENT_BINARY_DIR})

    #install(TARGETS simanneal RUNTIME DESTINATION ${CMAKE_BUILD_TYPE}/phys/simanneal)
    #install(FILES simanneal.physeng DESTINATION ${CMAKE_BUILD_TYPE}/phys/simanneal)
    install(TARGETS simanneal RUNTIME DESTINATION ${SIMANNEAL_INSTALL_DIR})
    install(FILES simanneal.physeng DESTINATION ${SIMANNEAL_INSTALL_DIR})

endif()
