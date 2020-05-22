#ifndef D95166DE_89E6_49FF_A7EA_BA27F7948D32
#define D95166DE_89E6_49FF_A7EA_BA27F7948D32

#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdint.h>
#include <boost/algorithm/string/split.hpp>
#include <algorithm>
#include <numeric>
#include <functional>
#include <endian.h>
#include <array>
#include <initializer_list>

#include "npy_array/endianess.h"
#include "npy_array/npy_exception.h"
#include "npy_array/npy_dtype.h"

template<typename T>
class npy_array
{
public:
    npy_array(const std::string& array_path);

    npy_array(const std::vector<size_t>& shape);
    npy_array(std::vector<size_t>&& shape);
    npy_array(std::initializer_list<size_t> shape_list);

    npy_array(const std::vector<size_t>& shape, const std::vector<T>& data);
    npy_array(std::vector<size_t>&& shape, std::vector<T>&& data);
    npy_array(std::initializer_list<size_t> shape_list, std::initializer_list<T> data_list);

    npy_array() = delete;
    npy_array(const npy_array& other) = default;
    npy_array(npy_array&& other) = default;

    ~npy_array() = default;

    npy_array& operator=(const npy_array& other) = default;
    npy_array& operator=(npy_array&& other) = default;
    T& operator[](size_t index);
    const T& operator[](size_t index) const;

    const std::vector<size_t>& shape() const noexcept;
    const npy_dtype& dtype() const noexcept;
    bool fortran_order() const noexcept;

    T* data() noexcept;
    const T* data() const noexcept;


    size_t size() const noexcept;
    size_t byte_size() const noexcept;

    void save(const std::string& array_path);
private:
    std::vector<size_t> _shape;
    std::vector<T> _data;
    npy_dtype _dtype;
    bool _fortran_order;

    bool check_magic_string(std::ifstream& array_file);
    bool check_version(std::ifstream& array_file);
    std::string read_header(std::ifstream& array_file);
    void parse_header(const std::string& header);
};

#include "npy_array/npy_array.ipp"

#endif /* D95166DE_89E6_49FF_A7EA_BA27F7948D32 */
