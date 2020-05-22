#include "npy_array/npy_exception.h"

npy_array_exception::npy_array_exception(const npy_array_exception_type exception_type) 
    : _exception_type{exception_type} {}

const char* npy_array_exception::what()
{
    switch (_exception_type)
        {
        case npy_array_exception_type::input_output_error:
            return "There has been an error while opening or reading the file.";
        case npy_array_exception_type::ill_formed_header:
            return "The header of the file is ill-formed.";
        case npy_array_exception_type::unsupported_version:
            return "The version of the file is unsupported.";
        case npy_array_exception_type::invalid_magic_string:
            return "The magic string is invalid.";
        case npy_array_exception_type::unsufficient_memory:
            return "The memory required is unsufficient.";
        case npy_array_exception_type::unsupported_dtype:
            return "The dtype required is not supported.";
        case npy_array_exception_type::unmatched_shape_data:
            return "The sizes of shape and data are mismatched.";
        case npy_array_exception_type::generic:
            return "There has been an error.";
        }
    return nullptr;
}

npy_array_exception_type npy_array_exception::exception_type() const
{
    return _exception_type;
}

const char* npy_dtype_exception::what()
{
    return "Invalid Dtype!";
}