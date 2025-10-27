file(GLOB VRU_SOURCES "*.cpp")
file(GLOB VRU_HEADERS "*.h")


# exclude files including tb_ from sources
list(FILTER VRU_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include_v3)

add_executable(sim_VRU_v3 testbench.cpp ${VRU_SOURCES} ${VRU_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")
