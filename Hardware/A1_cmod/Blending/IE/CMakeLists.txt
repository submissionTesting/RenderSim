file(GLOB IE_SOURCES "*.cpp")
file(GLOB IE_HEADERS "*.h")



# exclude files including tb_ from sources
list(FILTER IE_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include_v3)

add_executable(sim_IE testbench.cpp ${IE_SOURCES} ${IE_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")
