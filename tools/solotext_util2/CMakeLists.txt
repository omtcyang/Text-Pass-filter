cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(solotext_util2 LANGUAGES C CXX CUDA VERSION 1.0)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(PythonExtensions REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
find_library(TORCH_PYTHON_LIB NAME torch_python REQUIRED)


add_library(solotext_util_ext MODULE
    solotext_util2/src/registry.cu
    solotext_util2/src/get_text_masks.cu
)
python_extension_module(solotext_util_ext)

set_target_properties(solotext_util_ext PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(solotext_util_ext PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(solotext_util_ext ${TORCH_LIBRARIES} "c10_cuda" ${TORCH_PYTHON_LIB})
target_include_directories(solotext_util_ext PRIVATE ${TORCH_INCLUDE_DIRS})
set_property(TARGET solotext_util_ext PROPERTY CXX_STANDARD 14)

install(TARGETS solotext_util_ext DESTINATION solotext_util2)
