cmake_minimum_required(VERSION 3.10)

# set the project name
project(CLINGERUKF)
include(FetchContent)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -O2 or higher causes NaNs to be spit out
# this is due to the cholesky being weird
# -O1 is fast enough
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O1")

set(JSON_BuildTests OFF CACHE INTERNAL "")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(ukf ukf.cpp)
target_link_libraries(ukf PRIVATE Eigen3::Eigen)