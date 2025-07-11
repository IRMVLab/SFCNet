cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

option(SPCONV_BuildTests "Build the unit tests when BUILD_TESTING is enabled." ON)
option(SPCONV_BuildCUDA "Build cuda code when BUILD_TESTING is enabled." ON)
if (SPCONV_BuildCUDA)
    project(SparseConv LANGUAGES CXX CUDA VERSION 1.1)
else()
    project(SparseConv LANGUAGES CXX VERSION 1.1)
endif()

if(WIN32) # true if windows (32 and 64 bit)
    add_compile_definitions(TV_WINDOWS)
endif()

set(CMAKE_CXX_EXTENSIONS OFF) # avoid gnu++11 be added to CXX flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(TV_DEBUG)
endif()

find_package(Torch REQUIRED)
if (SPCONV_BuildCUDA)
    set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}")
    get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
    get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" DIRECTORY)
    if(WIN32) # true if windows (32 and 64 bit)
        set(CUDA_LIB_PATH_HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
    else()
        set(CUDA_LIB_PATH_HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
    endif()
    find_library(CUDA_CUDART NAMES cudart HINTS ${CUDA_LIB_PATH_HINTS})
    find_library(CUDA_CUBLAS NAMES cublas HINTS ${CUDA_LIB_PATH_HINTS})
    torch_cuda_get_nvcc_gencode_flag(NVCC_FLAGS_EXTRA)
    string (REPLACE ";" " " NVCC_FLAGS_EXTRA_STR "${NVCC_FLAGS_EXTRA}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS_EXTRA_STR}")
    add_compile_definitions(SPCONV_CUDA)
endif()

set(ALL_LIBS ${TORCH_LIBRARIES})

set(ALL_INCLUDE ${PROJECT_SOURCE_DIR}/include)



if (SPCONV_BuildCUDA)
    set(ALL_LIBS ${ALL_LIBS} ${CUDA_CUDART} ${CUDA_CUBLAS})
    set(ALL_INCLUDE ${ALL_INCLUDE} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    add_subdirectory(src/cuhash)
endif()
add_subdirectory(src/spproj)