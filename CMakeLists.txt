cmake_minimum_required(VERSION 3.13)
project(vol_decomposition C)

find_package(Python 3.13 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

Python_add_library(_vol_decomposition_c MODULE
    src/_c_src/vol_decomposition.c
    WITH_SOABI
)

target_include_directories(_vol_decomposition_c PRIVATE
    ${Python_NumPy_INCLUDE_DIRS}
)

target_compile_options(_vol_decomposition_c PRIVATE
    $<$<C_COMPILER_ID:MSVC>:/O2>
    $<$<NOT:$<C_COMPILER_ID:MSVC>>:-O3>
)

install(TARGETS _vol_decomposition_c DESTINATION src)
