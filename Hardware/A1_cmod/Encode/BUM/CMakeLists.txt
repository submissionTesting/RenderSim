file(GLOB BUM_SOURCES "*.cpp")
file(GLOB BUM_HEADERS "*.h")

list(FILTER BUM_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include_v5)

add_executable(sim_BUM testbench.cpp ${BUM_SOURCES} ${BUM_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")


