file(GLOB ROWGEN_SOURCES "*.cpp")
file(GLOB ROWGEN_HEADERS "*.h")

list(FILTER ROWGEN_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include_v6)

add_executable(sim_RowGeneration testbench.cpp ${ROWGEN_SOURCES} ${ROWGEN_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")


