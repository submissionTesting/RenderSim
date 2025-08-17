file(GLOB PE_ARRAY_SOURCES "*.cpp")
file(GLOB PE_ARRAY_HEADERS "*.h")



# exclude files including tb_ from sources
list(FILTER PE_ARRAY_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include ../PE/PE.h)

add_executable(sim_PE_array testbench.cpp ${PE_ARRAY_SOURCES} ${PE_ARRAY_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")
