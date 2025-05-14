cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0146 OLD)
cmake_policy(SET CMP0167 OLD)

project(cudaPCL LANGUAGES CXX CUDA)

# ----------------------------------------------------------------------------
# compiler settings
# ----------------------------------------------------------------------------
# enable C++14 and CUDA C++14
set(CMAKE_CXX_STANDARD     14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD       14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# mimic -D_REENTRANT, -fPIC, -O2
add_compile_options(
  "$<$<COMPILE_LANGUAGE:CXX>:-D_REENTRANT;-fPIC;-O2>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler;-fPIC;-O2;-lineinfo>"
)

# linker flags from LDFLAGS in Makefile
set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined -pthread"
)

# Tell CMake to compile for these compute capabilities
set(SM_ARCHS 30 35 37 50 52 53 60 61 62 70 72 75 87)
set(CMAKE_CUDA_ARCHITECTURES ${SM_ARCHS})

# ----------------------------------------------------------------------------
# include & link directories
# ----------------------------------------------------------------------------
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(VTK)
find_package(Boost REQUIRED COMPONENTS system)

set(VTK_INCLUDE_DIR "")
list(GET VTK_INCLUDE_DIRS 0 VTK_INCLUDE_DIR)
set(BUILD_EXAMPLE OFF CACHE BOOL "Activate the example building")

file(GLOB CUDAPCL_LIBRARIES "lib/*.so")
file(GLOB CUDAPCL_INCLUDE_DIRS "include")

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${CUDAPCL_INCLUDE_DIRS}
)

message(STATUS "Libraries: ${CUDAPCL_LIBRARIES}")
message(STATUS "Include dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "BUILD_EXAMPLE: ${BUILD_EXAMPLE}")

if(BUILD_EXAMPLE)
   add_executable(cluster_cuda examples/cluster_main.cpp)
   target_link_libraries(cluster_cuda ${CUDA_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES} ${CUDAPCL_LIBRARIES})
   
   add_executable(filter_cuda examples/filter_main.cpp)
   target_link_libraries(filter_cuda ${CUDA_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES} ${CUDAPCL_LIBRARIES})
   
   add_executable(icp_cuda examples/icp_main.cpp)
   target_link_libraries(icp_cuda ${CUDA_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES} ${CUDAPCL_LIBRARIES})
   
   add_executable(ndt_cuda examples/ndt_main.cpp)
   target_link_libraries(ndt_cuda ${CUDA_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES} ${CUDAPCL_LIBRARIES})
   
   add_executable(octree_cuda examples/octree_main.cpp)
   target_link_libraries(octree_cuda ${CUDA_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES} ${CUDAPCL_LIBRARIES})
   
   add_executable(segmentation_cuda examples/segmentation_main.cpp)
   target_link_libraries(segmentation_cuda ${CUDA_LIBRARIES} ${PCL_LIBRARIES} ${VTK_LIBRARIES} ${CUDAPCL_LIBRARIES})
endif()

# ----------------------------------------------------------------------------
# finish
# ----------------------------------------------------------------------------
message(STATUS "Configured cudaPCL with CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")

