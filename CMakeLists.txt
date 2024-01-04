cmake_minimum_required(VERSION 3.11)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12")
    cmake_policy(SET CMP0074 NEW)
endif()

# for release builds, add flag "-DCMAKE_BUILD_TYPE=Release" (without quotes) when running cmake.

project(simanneal CXX)
# TODO add way to push version number into source code

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

    # set_target_properties(libraryname PROPERTIES LINK_FLAGS "-Wl,-znodelete")
    
    find_package(SWIG REQUIRED)
    include(${SWIG_USE_FILE})

    find_package(PythonLibs REQUIRED)
    include_directories(${PYTHON_INCLUDE_PATH})

    set_property(SOURCE swig/simanneal.i PROPERTY CPLUSPLUS ON)
    file(COPY swig/__init__.py DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/swig")

    swig_add_library(simanneal LANGUAGE python SOURCES swig/simanneal.i src/simanneal.cc src/global.cc)

    if (WIN32 OR APPLE)
        swig_link_libraries(simanneal ${Boost_LIBRARIES} ${PYTHON_LIBRARY})
    else()
        swig_link_libraries(simanneal ${Boost_LIBRARIES})
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
    )

    set(SIMANNEAL_TESTS
        tests/simanneal.test.cc

        # HEADERS
        libs/catch2/catch.hpp
    )

    add_library(simanneal_lib STATIC ${SIMANNEAL_SOURCES})
    target_link_libraries(simanneal_lib siqadconn ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

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

        add_dependencies(simanneal_tests simanneal)

        set_property(TARGET simanneal_tests PROPERTY CXX_STANDARD 11)
        set_property(TARGET simanneal_tests PROPERTY CXX_STANDARD_REQUIRED True)

        enable_testing()
        add_test(simanneal_tests simanneal_tests)
        # add_custom_command(
        #     TARGET simanneal_tests
        #     POST_BUILD
        #     COMMAND ctest -C $<CONFIGURATION> --output-on-failure --verbose
        # )
    endif()

    install(TARGETS simanneal RUNTIME DESTINATION ${SIMANNEAL_INSTALL_DIR})
    install(FILES simanneal.physeng ubc_logo.png DESTINATION ${SIMANNEAL_INSTALL_DIR})

endif()
