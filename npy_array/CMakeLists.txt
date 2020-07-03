find_package(Boost 1.72.0  COMPONENTS regex)

set(NPY_ARRAY_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/npy_array" CACHE STRING "NPY Array library include path")
set(NPY_ARRAY_LIB_NAME "npy_array" CACHE STRING "NPY Array library name")


set(NPY_ARRAY_SOURCES npy_dtype.cpp npy_exception.cpp)
set(BOOST_LIB_NAMES boost_regex)

add_library(npy_array SHARED "${NPY_ARRAY_SOURCES}")
add_library(npy_array_static STATIC "${NPY_ARRAY_SOURCES}")

target_link_libraries(npy_array "${BOOST_LIB_NAMES}")
target_link_libraries(npy_array_static "${BOOST_LIB_NAMES}")

target_link_libraries(npy_array Boost::regex)
target_link_libraries(npy_array_static Boost::regex)

target_include_directories(npy_array PUBLIC "${Boost_INCLUDE_DIRS}")
target_include_directories(npy_array_static PUBLIC "${Boost_INCLUDE_DIRS}")