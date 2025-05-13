cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0146 OLD)
cmake_policy(SET CMP0167 OLD)

project(demo LANGUAGES CXX CUDA)

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

# ----------------------------------------------------------------------------
# CUDA architectures (GENCODE_FLAGS → CMAKE_CUDA_ARCHITECTURES)
# ----------------------------------------------------------------------------
set(SM_ARCHS 30 35 37 50 52 53 60 61 62 70 72 75 87)
# Tell CMake to compile for these compute capabilities
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

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  include
)

link_directories(lib)

add_executable(${PROJECT_NAME} 
src/main.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
    ${PCL_LIBRARIES}
    ${VTK_LIBRARIES}
    libcudacluster.so
)

# ----------------------------------------------------------------------------
# finish
# ----------------------------------------------------------------------------
message(STATUS "Configured demo with CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")

