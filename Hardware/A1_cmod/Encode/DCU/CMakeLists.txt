file(GLOB DCU_SOURCES "*.cpp")
file(GLOB DCU_HEADERS "*.h")



# exclude files including tb_ from sources
list(FILTER DCU_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include)

add_executable(sim_DCU testbench.cpp ${DCU_SOURCES} ${DCU_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")
