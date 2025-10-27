cmake_minimum_required(VERSION 3.22)
project(SRAMTest)

set(CMAKE_CXX_STANDARD 11)

# Check for CATAPULT_HOME
if(NOT DEFINED ENV{CATAPULT_HOME})
    message(FATAL_ERROR "Environment variable CATAPULT_HOME is not set.")
endif()

# Check for MATCHLIB_HOME
if(NOT DEFINED ENV{MATCHLIB_HOME})
    message(FATAL_ERROR "Environment variable MATCHLIB_HOME is not set.")
endif()

set(CATAPULT_HOME $ENV{CATAPULT_HOME})
set(MATCHLIB_HOME $ENV{MATCHLIB_HOME})

# set catapult compiler
set(CMAKE_C_COMPILER ${CATAPULT_HOME}/Mgc_home/pkgs/dcs_gcc/gcc-10.3.0/bin/gcc)
set(CMAKE_CXX_COMPILER ${CATAPULT_HOME}/Mgc_home/pkgs/dcs_gcc/gcc-10.3.0/bin/g++)

# matchlib nvlibs
include_directories(${MATCHLIB_HOME})

# catapult systemc libs
include_directories(${CATAPULT_HOME}/Mgc_home/shared/include)
link_directories(${CATAPULT_HOME}/Mgc_home/shared/lib)
link_libraries(${CATAPULT_HOME}/Mgc_home/shared/lib/libsystemc.a)

link_directories(${CATAPULT_HOME}/Mgc_home/pkgs/dcs_gcc/gcc-10.3.0/lib64)
link_libraries(${CATAPULT_HOME}/Mgc_home/pkgs/dcs_gcc/gcc-10.3.0/lib64/libstdc++.a)

# Boost (optional)
FIND_PACKAGE(Boost)
IF (Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    ADD_DEFINITIONS( "-DHAS_BOOST" )
ENDIF()

# shared compilation options (mirror Blending)
set(CFLAGS "-Wall -Wno-unknown-pragmas -DDEBUG_LEVEL=1")
set(LIBFLAGS "-lstdc++ -lsystemc -lm -lpthread -lboost_timer -lboost_chrono -lboost_system")
set(SIMFLAGS "-DCONNECTIONS_ACCURATE_SIM -DSC_INCLUDE_DYNAMIC_PROCESSES")

file(GLOB SRAMTEST_SOURCES "*.cpp")
file(GLOB SRAMTEST_HEADERS "*.h")

list(FILTER SRAMTEST_SOURCES EXCLUDE REGEX "testbench.*\\.cpp")

include_directories(./)
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/..)

add_executable(sim_SRAMTest testbench.cpp ${SRAMTEST_SOURCES} ${SRAMTEST_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")
