cmake_minimum_required(VERSION 3.0.0)
project(imslic_project VERSION 0.1.0)

set(CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")

add_subdirectory(npy_array)
add_subdirectory(thread_pool)
add_subdirectory(imslic)

add_executable(imslic_exe imslic.cpp)
target_link_libraries(imslic_exe "${IMSLIC_LIB_NAME}" "${NPY_ARRAY_LIB_NAME}" "${THREAD_POOL_LIB_NAME}")
target_include_directories(imslic_exe PUBLIC "${IMSLIC_INCLUDE_PATH}" "${NPY_ARRAY_INCLUDE_PATH}" "${THREAD_POOL_INCLUDE_PATH}")

add_custom_command(
        TARGET imslic_exe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/image.npy
                ${CMAKE_CURRENT_BINARY_DIR}/image.npy
)

add_custom_command(
        TARGET imslic_exe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/4k.npy
                ${CMAKE_CURRENT_BINARY_DIR}/4k.npy
)