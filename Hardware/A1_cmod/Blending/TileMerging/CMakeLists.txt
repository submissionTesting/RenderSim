file(GLOB TILEMERGING_SOURCES "*.cpp")
file(GLOB TILEMERGING_HEADERS "*.h")

list(FILTER TILEMERGING_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include_v6)

add_executable(sim_TileMerging testbench.cpp ${TILEMERGING_SOURCES} ${TILEMERGING_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")


