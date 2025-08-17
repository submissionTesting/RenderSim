file(GLOB PE_SOURCES "*.cpp")
file(GLOB PE_HEADERS "*.h")



# exclude files including tb_ from sources
list(FILTER PE_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include)

add_executable(sim_PE testbench.cpp ${PE_SOURCES} ${PE_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")
