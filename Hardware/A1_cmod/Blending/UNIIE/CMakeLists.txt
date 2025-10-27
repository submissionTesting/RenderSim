file(GLOB UNIIE_SOURCES "*.cpp")
file(GLOB UNIIE_HEADERS "*.h")



# exclude files including tb_ from sources
list(FILTER UNIIE_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include_v3)

add_executable(sim_UNIIE testbench.cpp ${UNIIE_SOURCES} ${UNIIE_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")
