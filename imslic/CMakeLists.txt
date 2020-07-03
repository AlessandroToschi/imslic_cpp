set(IMSLIC_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/imslic" CACHE STRING "Imslic library include path")
set(IMSLIC_LIB_NAME "imslic" CACHE STRING "NPY Array library name")


set(IMSLIC_SOURCES rgb_to_lab.cpp)

add_library(imslic SHARED "${IMSLIC_SOURCES}")
add_library(imslic_static STATIC "${IMSLIC_SOURCES}")

target_link_libraries(imslic "${NPY_ARRAY_LIB_NAME}" "${THREADPOOL_LIB_NAME}")
target_link_libraries(imslic_static "${NPY_ARRAY_LIB_NAME}" "${THREADPOOL_LIB_NAME}")

target_include_directories(imslic PUBLIC "${NPY_ARRAY_INCLUDE_PATH}" "${THREADPOOL_INCLUDE_PATH}")
target_include_directories(imslic_static PUBLIC "${NPY_ARRAY_INCLUDE_PATH}" "${THREADPOOL_INCLUDE_PATH}")