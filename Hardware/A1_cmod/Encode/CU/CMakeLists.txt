file(GLOB CU_SOURCES "*.cpp")
file(GLOB CU_HEADERS "*.h")



# exclude files including tb_ from sources
list(FILTER CU_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include_v4)

add_executable(sim_CU testbench.cpp ${CU_SOURCES} ${CU_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")
