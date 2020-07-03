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

#include "endianess.h"
#include "npy_exception.h"
#include "npy_dtype.h"

template<typename T>
class npy_array
{
public:
    typedef T value_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef size_t size_type;
    typedef T* iterator;
    typedef const T* const_iterator;

    npy_array(const std::string& array_path);

    npy_array(const std::vector<size_type>& shape);
    npy_array(std::vector<size_type>&& shape);
    npy_array(std::initializer_list<size_type> shape_list);

    npy_array(const std::vector<size_type>& shape, const std::vector<T>& data);
    npy_array(std::vector<size_type>&& shape, std::vector<T>&& data);
    npy_array(std::initializer_list<size_type> shape_list, std::initializer_list<T> data_list);

    npy_array() = delete;
    npy_array(const npy_array& other) = default;
    npy_array(npy_array&& other) = default;

    ~npy_array() = default;

    npy_array& operator=(const npy_array& other) = default;
    npy_array& operator=(npy_array&& other) = default;

    reference operator[](size_type index) noexcept;
    const_reference operator[](size_type index) const noexcept;

    reference operator[](std::initializer_list<size_type> indexes) noexcept;
    const_reference operator[](std::initializer_list<size_type> indexes) const noexcept;

    reference at(size_type index);
    const_reference at(size_type index) const;

    reference at(std::initializer_list<size_type> indexes);
    const_reference at(std::initializer_list<size_type> indexes) const;

    const std::vector<size_type>& shape() const noexcept;
    const npy_dtype& dtype() const noexcept;
    bool fortran_order() const noexcept;

    iterator begin() noexcept;
    const_iterator begin() const noexcept;
    const_iterator cbegin() const noexcept;

    iterator end() noexcept;
    const_iterator end() const noexcept; 
    const_iterator cend() const noexcept;

    T* data() noexcept;
    const T* data() const noexcept;

    size_type size() const noexcept;
    size_type byte_size() const noexcept;

    void save(const std::string& array_path);
private:
    std::vector<size_type> _shape;
    std::vector<T> _data;
    std::vector<size_type> _strides;
    npy_dtype _dtype;
    bool _fortran_order;

    bool check_magic_string(std::ifstream& array_file);
    bool check_version(std::ifstream& array_file);
    std::string read_header(std::ifstream& array_file);
    void parse_header(const std::string& header);
    void check_for_strides();
};

#include "npy_array.ipp"

#endif /* D95166DE_89E6_49FF_A7EA_BA27F7948D32 */
