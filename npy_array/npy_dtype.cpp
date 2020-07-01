#include "npy_dtype.h"

#include <iostream>

// Initialize the null dtype with unknown kind, item size 0, and endianess not applicable.
npy_dtype::npy_dtype() noexcept 
    : _kind{npy_dtype_kind::unkwown}, _item_size{0}, _byte_order{npy_endianness::not_applicable} {}

// Initialize the attributes with the ones provided by the user.
// No checks are performed for the given input parameters.
npy_dtype::npy_dtype(npy_dtype_kind kind, size_t item_size, npy_endianness byte_order) noexcept
    : _kind{kind}, _item_size{item_size}, _byte_order{byte_order} {}

// Move constructor, copy the attributes from the other dtype and then make the other identical to the null dtype.
npy_dtype::npy_dtype(npy_dtype&& other) 
    : _kind{std::move(other._kind)}, _item_size{std::move(other._item_size)}, _byte_order{std::move(other._byte_order)}
{
    if(this != &other)
    {
        // Make the other equals to null dtype.
        other._kind = npy_dtype_kind::unkwown;
        other._item_size = 0;
        other._byte_order = npy_endianness::not_applicable;
    }
}

// Move assignment.
npy_dtype& npy_dtype::operator=(npy_dtype&& other)
{
    if(this != &other)
    {
        // Move the attributes from the other dtype.
        _kind = std::move(other._kind);
        _item_size = std::move(other._item_size);
        _byte_order = std::move(other._byte_order);

        // Make the other equals to null dtype.
        other._kind = npy_dtype_kind::unkwown;
        other._item_size = 0;
        other._byte_order = npy_endianness::not_applicable;
    }

    return *this;
}

npy_dtype::operator bool() const noexcept
{
    return (*this != npy_dtype::null());
}

bool npy_dtype::operator==(const npy_dtype& other) const noexcept
{
    return _kind == other._kind && _item_size == other._item_size && _byte_order == other._byte_order;
}

bool npy_dtype::operator!=(const npy_dtype& other) const noexcept
{
    return !(*this == other);
}

npy_dtype_kind npy_dtype::kind() const {return _kind;} // Return the private npy_dtype_kind.
size_t npy_dtype::item_size() const {return _item_size;} // Return the private item size.
npy_endianness npy_dtype::byte_order() const {return _byte_order;} // Return the private npy_endianess.

std::string npy_dtype::str() const
{
    // Initialize the dtype string as empty and then push in order the byte order (endianess),
    // the kind, and the item size converted to string.
    std::string dtype_string{};
    dtype_string.push_back(static_cast<char>(_byte_order));
    dtype_string.push_back(static_cast<char>(_kind));
    dtype_string.append(std::to_string(_item_size));
    return dtype_string;
}

npy_dtype npy_dtype::from_string(const std::string& dtype_string) noexcept
{
    // Regex pattern for matchning dtype string format.
    // The first (index 0) group is always the full matched string.
    // The second (index 1) group is a literal match of 1 byte types as "|b1", "|i1", and "|u1".
    // If the pattern is not a one byte type, then:
    // The third (index 2) group is the endianess, which is optional because otherwise it assumed to be the native one.
    // The fourth (index 3) group is the kind, which is non optional.
    // The fifth (index 4) group is the item size, which is optional because otherwise it will picked a default item size.
    boost::regex dtype_pattern{R"((\|b1|\|u1|\|i1)|(^[<>=]?)((?<!\|)[iufc])((?<=[iu])2|(?<=[iuf])4|(?<=[uifc])8|(?<=[fc])16|(?<=[c])32)?$)"};
    boost::match_results<std::string::const_iterator> match_results{}; // Where the matching results of the regex are stored.

    npy_dtype_kind kind; // The temporary kind variable used to create the npy_dtype object.
    size_t item_size; // The temporary item size variable used to create the npy_dtype object.
    npy_endianness byte_order = get_endianess(); // The temporary endianess variable used to create the npy_dtype object, by default use native endianess.

    // Search the pattern along the whole string.
    if(boost::regex_search(dtype_string.cbegin(), dtype_string.cend(), match_results, dtype_pattern))
    { 
        // The match size must be 5, otherwise something happened and we will return the null dtype.
        if(match_results.size() == 5)
        {
            // If the second group is matched means the user wants a one byte type.
            // We can directly return from this block by returning the appropriate one byte dtype.
            // If something went wrong then return a null dtype.
            if(match_results[1].matched)
            {
                auto one_byte_dtype = match_results[1].str();

                // Hard-coded comparison
                if(one_byte_dtype == "|b1") return npy_dtype::bool_8();
                else if(one_byte_dtype == "|i1") return npy_dtype::int_8();
                else if(one_byte_dtype == "|u1") return npy_dtype::uint_8();
                else return npy_dtype::null();
            }

            // If the third group is matched means that the user has specified an endianess for the dtype.
            if(match_results[2].matched)
            {
                // Since npy_endianess is an enum backed by a char, we can statically cast the char to enum safely.
                npy_endianness requested_endianess = static_cast<npy_endianness>(match_results[2].str()[0]);

                // If the requested byte order is native, then the variable is already initialized, otherwise we store what the user has requested as byte order.
                if(requested_endianess != npy_endianness::native) byte_order = requested_endianess;
            }

            // If the fourth group is matched means that the user has specified the dtype kind.
            // This field is mandatory, if something goes wrong then we return a null dtype.
            if(match_results[3].matched) kind = static_cast<npy_dtype_kind>(match_results[3].str()[0]);
            else return npy_dtype::null();

            // If the fifth group is matched means that the user has specified the item size.
            // The item size is a string and it is converted to a size type.
            // No conversion errors are possible because the regex pattern allows only predefined size.
            // If the fifth group is not matched than provide a default item size that is 8 bytes for integer, either signed or unsigned, and floating points, and 16 bytes for the complex type.
            if(match_results[4].matched) item_size = std::stoul(match_results[4].str());
            else
            {
                switch (kind)
                {
                case npy_dtype_kind::integer:
                case npy_dtype_kind::floating_point:
                case npy_dtype_kind::not_signed:
                    item_size = 8;
                    break;
                case npy_dtype_kind::complex:
                    item_size = 16;
                    break;
                default:
                    return npy_dtype::null();
                }
            }
            // All the temporary variables are initialized and checked, ready to return the wanted type.
            return npy_dtype(kind, item_size, byte_order);
        }
    }
    // In any case, return the null dtype.
    return npy_dtype::null();
}

npy_dtype npy_dtype::bool_8() noexcept {return npy_dtype{npy_dtype_kind::boolean, sizeof(bool), npy_endianness::not_applicable};}
npy_dtype npy_dtype::int_8() noexcept {return npy_dtype{npy_dtype_kind::integer, sizeof(int8_t),npy_endianness::not_applicable};}
npy_dtype npy_dtype::int_16() noexcept {return npy_dtype{npy_dtype_kind::integer, sizeof(int16_t)};}
npy_dtype npy_dtype::int_32() noexcept {return npy_dtype{npy_dtype_kind::integer, sizeof(int32_t)};}
npy_dtype npy_dtype::int_64() noexcept {return npy_dtype{npy_dtype_kind::integer, sizeof(int64_t)};}
npy_dtype npy_dtype::uint_8() noexcept {return npy_dtype{npy_dtype_kind::not_signed, sizeof(uint8_t), npy_endianness::not_applicable};}
npy_dtype npy_dtype::uint_16() noexcept {return npy_dtype{npy_dtype_kind::not_signed, sizeof(uint16_t)};}
npy_dtype npy_dtype::uint_32() noexcept {return npy_dtype{npy_dtype_kind::not_signed, sizeof(uint32_t)};}
npy_dtype npy_dtype::uint_64() noexcept {return npy_dtype{npy_dtype_kind::not_signed, sizeof(uint64_t)};}
npy_dtype npy_dtype::float_32() noexcept {return npy_dtype{npy_dtype_kind::floating_point, sizeof(float)};}
npy_dtype npy_dtype::float_64() noexcept {return npy_dtype{npy_dtype_kind::floating_point, sizeof(double)};}
npy_dtype npy_dtype::float_128() noexcept {return npy_dtype{npy_dtype_kind::floating_point, sizeof(long double)};}
npy_dtype npy_dtype::complex_64() noexcept {return npy_dtype{npy_dtype_kind::complex, sizeof(std::complex<float>)};}
npy_dtype npy_dtype::complex_128() noexcept {return npy_dtype{npy_dtype_kind::complex, sizeof(std::complex<double>)};}
npy_dtype npy_dtype::complex_256() noexcept {return npy_dtype{npy_dtype_kind::complex, sizeof(std::complex<long double>)};}
npy_dtype npy_dtype::null() noexcept {static npy_dtype null_dtype{}; return null_dtype;}