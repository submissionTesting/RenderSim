file(GLOB FRM_SOURCES "*.cpp")
file(GLOB FRM_HEADERS "*.h")

list(FILTER FRM_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include_v5)

add_executable(sim_FRM testbench.cpp ${FRM_SOURCES} ${FRM_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")


