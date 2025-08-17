cmake_minimum_required(VERSION 3.14)
project(rendersim_scheduler LANGUAGES CXX)

# C++17 is plenty
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# Core library (placeholder)
# -----------------------------------------------------------------------------
add_library(rendersim_scheduler STATIC
    src/ir.cpp
    src/optimization_library.cpp
    src/operator_scheduler.cpp
    src/system_scheduler.cpp
    src/ppa_estimator.cpp
    src/ramulator_interface.cpp
    # Mapping loader (C++ implementation of IR JSON parser)
    src/mapping/mapping_loader.cpp
    # HW config loader and C++ mapping engine
    src/mapping/hw_config_loader.cpp
    src/mapping/mapping_engine.cpp
)

target_include_directories(rendersim_scheduler PUBLIC include)

# Optional: link nlohmann_json if available (header-only library)
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    target_link_libraries(rendersim_scheduler PUBLIC nlohmann_json::nlohmann_json)
endif()

# Fallback: if header exists in local Conda pkgs directory packaged with repo, add it to include paths
if(EXISTS "${CMAKE_SOURCE_DIR}/log/RenderSim/pkgs/nlohmann_json-3.11.2-h6a678d5_0/include/nlohmann/json.hpp")
    message(STATUS "Using vendored nlohmann_json headers from log/RenderSim/pkgs")
    target_include_directories(rendersim_scheduler PUBLIC "${CMAKE_SOURCE_DIR}/log/RenderSim/pkgs/nlohmann_json-3.11.2-h6a678d5_0/include")
endif()

# Enable position-independent code for shared library linking
set_target_properties(rendersim_scheduler PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# Python bindings via pybind11 (optional, build if pybind11 is found)
# -----------------------------------------------------------------------------
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.tar.gz
    )
    FetchContent_MakeAvailable(pybind11)
    set(pybind11_FOUND ON)
endif()

if(pybind11_FOUND)
    pybind11_add_module(rendersim_cpp MODULE src/bindings.cpp)
    target_link_libraries(rendersim_cpp PRIVATE rendersim_scheduler)
endif() 