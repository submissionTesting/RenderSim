file(GLOB CCU_SOURCES "*.cpp")
file(GLOB CCU_HEADERS "*.h")

list(FILTER CCU_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include)

add_executable(sim_CullingConversionUnit testbench.cpp ${CCU_SOURCES} ${CCU_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")


