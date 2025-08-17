file(GLOB HAMAT_SOURCES "*.cpp")
file(GLOB HAMAT_HEADERS "*.h")



# exclude files including tb_ from sources
list(FILTER HAMAT_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include)

add_executable(sim_BSU testbench.cpp ${HAMAT_SOURCES} ${HAMAT_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")
