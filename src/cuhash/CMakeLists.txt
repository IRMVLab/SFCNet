
add_library(cuhash STATIC hash_functions.cu hash_table.cpp hash_table.cu hash_functions.cpp)

target_include_directories(cuhash PRIVATE ${ALL_INCLUDE})
set_property(TARGET cuhash PROPERTY CUDA_STANDARD 14)
set_property(TARGET cuhash PROPERTY CXX_STANDARD 14)
set_target_properties(cuhash PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuhash PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if(NOT WIN32)
    set_property(TARGET cuhash PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
target_link_libraries(cuhash PRIVATE ${ALL_LIBS})