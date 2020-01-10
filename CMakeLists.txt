cmake_minimum_required(VERSION 3.10)

# for release builds, add flag "-DCMAKE_BUILD_TYPE=Release" (without quotes) when running cmake.

project(simanneal CXX)
# TODO add way to push version number into source code

find_package(Boost COMPONENTS system filesystem thread REQUIRED)
find_package(Threads REQUIRED)

#add_definitions( -DAPP_VERSION=\"0.1.0\" -DAPPLICATION_NAME=\"SiQAD\" -DORGANIZATION_NAME=\"WalusLab\" )

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(SOURCES
    main.cc
    global.cc
    interface.cc
    sim_anneal.cc
    siqadconn.cc
)

include_directories(
    .
    ${Boost_INCLUDE_DIR}
)

add_executable(simanneal ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
