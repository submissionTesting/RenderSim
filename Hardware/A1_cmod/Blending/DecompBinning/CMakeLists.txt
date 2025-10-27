file(GLOB DBIN_SOURCES "*.cpp")
file(GLOB DBIN_HEADERS "*.h")

list(FILTER DBIN_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include_v6)

add_executable(sim_DecompBinning testbench.cpp ${DBIN_SOURCES} ${DBIN_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")


