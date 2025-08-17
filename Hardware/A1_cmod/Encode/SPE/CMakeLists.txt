file(GLOB SPE_SOURCES "*.cpp")
file(GLOB SPE_HEADERS "*.h")



# exclude files including tb_ from sources
list(FILTER SPE_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include)
include_directories(../CU)
include_directories(../DCU)

add_executable(sim_SPE testbench.cpp ${SPE_SOURCES} ${SPE_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")
