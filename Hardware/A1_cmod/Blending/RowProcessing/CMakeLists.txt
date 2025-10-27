file(GLOB ROWPROC_SOURCES "*.cpp")
file(GLOB ROWPROC_HEADERS "*.h")

list(FILTER ROWPROC_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include_v6)

add_executable(sim_RowProcessing testbench.cpp ${ROWPROC_SOURCES} ${ROWPROC_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")


