file(GLOB RFM_SOURCES "*.cpp")
file(GLOB RFM_HEADERS "*.h")



# exclude files including tb_ from sources
list(FILTER RFM_SOURCES EXCLUDE REGEX "tb_")

include_directories(../include)

add_executable(sim_RFM testbench.cpp ${RFM_SOURCES} ${RFM_HEADERS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CFLAGS} ${SIMFLAGS} ${LIBFLAGS}")
