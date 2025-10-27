file(GLOB SORTING_SOURCES "*.cpp")
file(GLOB SORTING_HEADERS "*.h")

list(FILTER SORTING_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include_v6)

add_executable(sim_Sorting testbench.cpp ${SORTING_SOURCES} ${SORTING_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")


