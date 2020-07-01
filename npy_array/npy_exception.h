#ifndef E54F2198_A9A7_41DA_8A9B_8950CF35D709
#define E54F2198_A9A7_41DA_8A9B_8950CF35D709

#include <exception>

enum npy_array_exception_type
{
    input_output_error,
    ill_formed_header,
    unsupported_version,
    invalid_magic_string,
    unsufficient_memory,
    unsupported_dtype,
    unmatched_shape_data,
    generic
};

class npy_array_exception : std::exception
{
public:
    npy_array_exception(const npy_array_exception_type exception_type);

    const char* what();

    npy_array_exception_type exception_type() const;

private:
    npy_array_exception_type _exception_type;
};

class npy_dtype_exception : std::exception
{
public:
    const char* what();
};
#endif /* E54F2198_A9A7_41DA_8A9B_8950CF35D709 */
