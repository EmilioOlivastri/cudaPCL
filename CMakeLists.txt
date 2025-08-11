cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0074 OLD)

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

file(GLOB_RECURSE CUDAPCL_TF_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_transform.cpp
  PROPERTIES LANGUAGE CUDA
)

add_library(CUDAPCL_TF_LIB ${CUDAPCL_TF_SOURCES})

set_target_properties(CUDAPCL_TF_LIB PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(CUDAPCL_TF_LIB
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(CUDAPCL_TF_LIB PUBLIC
    ${CUDA_LIBRARIES}
    Boost::system
)


file(GLOB CUDAPCL_LIBRARIES "lib/*.so")
file(GLOB CUDAPCL_INCLUDE_DIRS "include")
list(APPEND CUDAPCL_LIBRARIES CUDAPCL_TF_LIB)
set(CUDAPCL_LIBRARIES ${CUDAPCL_LIBRARIES} PARENT_SCOPE)
set(CUDAPCL_INCLUDE_DIRS ${CUDAPCL_INCLUDE_DIRS} PARENT_SCOPE) 

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${CUDAPCL_INCLUDE_DIRS}
)

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

