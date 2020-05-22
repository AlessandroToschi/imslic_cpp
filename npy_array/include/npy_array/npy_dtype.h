#ifndef D4AB99B8_10DA_46D7_9CFA_559B460F6ABA
#define D4AB99B8_10DA_46D7_9CFA_559B460F6ABA

#include <cstddef>
#include <type_traits>
#include <complex>

#include <boost/regex.hpp>

#include "npy_array/endianess.h"
#include "npy_array/npy_exception.h"

/**
 * @brief Enumerator that describes the possible types of a dtype.
 * 
 * Only native dtytes are supported: boolean, integers, unsigned integers, floating points, and complex.
 * The structured dtypes are not supported.
 * Each kind is backed by a char that uniquely identifies the kind and it is part of the dtype string format.
 * The 'unkown' kind is an artificial kind used to express a not valid kind, it is used by the null dtype.
 * 
 */
enum npy_dtype_kind
{
    boolean = 'b',
    integer = 'i',
    not_signed = 'u', // unsigned integers.
    floating_point = 'f',
    complex = 'c',
    unkwown = '!' // artificial kind used for the null dtype.
};

/**
 * @brief Class that expresses a NumPy dtype object.
 * 
 * A dtype object is identified by three attributes:
 * kind: the kind of the type.
 * item size: the size in bytes of the dtype object.
 * byte order: the endianess of the dtype object.
 * 
 * The byte order applies only for an dtype object with item size >= 2.
 * For an dtype object with item size 1, the byte order is not applicable.
 * 
 * A npy_dtype object is immutable and read-only.
 * The use can create a npy_dtype using two static methods:
 * 
 * npy_dtype::from_type<T>(), which creates the npy_dtype given the C++ type.
 * If the C++ type is not one of the implemented, then it is returned the null dtype.
 * 
 * npy_dtype::from_string(dtype_string), which creates the npy_dtype given a NumPy dtype object string format.
 * If the dtype_string is not one of the implemented or malformed, then it is returned the null dtype.
 * 
 * Alternatively, the user can select one of the native types provided as static methods.
 * 
 * The implemented types are:
 * bool_8, 8-bit boolean value backed by C++ bool type.
 * int_8, signed 8-bit integer value backed by C++ int8_t and char.
 * int_16, signed 16-bit integer value backed by C++ int16_t and short.
 * int_32, signed 32-bit integer value backed by C++ int32_t and int.
 * int_64, signed 64-bit integer value backed by C++ int64_t and long.
 * uint_8, unsigned 8-bit integer value backed by C++ uint8_t and unsigned char.
 * uint_16, unsigned 16-bit integer value backed by C++ uint16_t and unsigned short.
 * uint_32, unsigned 32-bit integer value backed by C++ uint32_t and unsigned int.
 * uint_64, unsigned 64-bit integer value backed by C++ uint64_t and unsigned long.
 * float_32, 32-bit floating point value backed by C++ float.
 * float_64, 64-bit floating point value backed by C++ double.
 * float_128, 128-bit floating point value backed by C++ long double.
 * complex_64, 64-bit complex value backed by C++ std::complex<float>.
 * complex_128, 128-bit complex value backed by C++ std::complex<double>.
 * complex_256, 256-bit complex value backed by C++ std::complex<long double>.
 * 
 * The whole object does not throw exceptions, but when errors occur (during the creation of an object) the null dtype is returned.
 * If the user builds its own dtype, either by providing a type or a dtype string, it must check if the retrieved dtype is not equal to the null dtype.
 * A null dtype has the following invariants:
 * kind = npy_dtype_kind::unknown
 * item size = 0
 * byte order = npy_endianess::not_applicable.
 */
class npy_dtype
{
public:
    /**
     * @brief Construct a null npy_dtype object.
     * 
     * The null dtype object has the following invariants:
     * kind = npy_dtype_kind::unknown
     * item size = 0
     * byte order = npy_endianess::not_applicable.>
     */
    npy_dtype() noexcept;

    /**
     * @brief Copy constructor, construct a dtype object given another one.
     * 
     * The copy constructor is compiler-generated.
     * 
     * @param other another dtype object.
     */
    npy_dtype(const npy_dtype& other) = default;
    /**
     * @brief Move constructor, construct a new dtype object by moving another one.
     * 
     * The move constructor leaves the other object identical to the null dtype.
     * 
     * @param other another dtype object.
     */
    npy_dtype(npy_dtype&& other);

    /**
     * @brief Destroy the npy_dtype object
     * 
     * Compiler-generated, this class contains only builtin types and no dynamic memory is involed.
     * 
     */
    ~npy_dtype() = default;
    
    /**
     * @brief Copy assignment, copy the given dtype object to this.
     * 
     * The copy assignment is compiler-generated.
     * 
     * @param other the dtype object to copy.
     * @return npy_dtype& this object.
     */
    npy_dtype& operator=(const npy_dtype& other) = default;
    /**
     * @brief Move assignment, move the given dtype object to this.
     * 
     * The move assignment leaves the other object identical to the null dtype.
     * 
     * @param other the dtype object to move.
     * @return npy_dtype& this object.
     */
    npy_dtype& operator=(npy_dtype&& other);


    explicit operator bool() const noexcept;

    bool operator==(const npy_dtype& other) const noexcept;
    bool operator!=(const npy_dtype& other) const noexcept;

    /**
     * @brief The kind of this dtype object.
     * 
     * Available kinds:
     * boolean, intgers, unsigned integers, floating points, and complex.
     * 
     * @return npy_dtype_kind the kind of this dtype object.
     */
    npy_dtype_kind kind() const;
    /**
     * @brief The size in bytes taken by an element having this dtype.
     * 
     * A valid dtype has a size always >= 0.
     * The null dtype has a size of 0.
     * 
     * @return size_t the size in bytes.
     */
    size_t item_size() const;
    /**
     * @brief The byte order, or endianess, of this dtype when the item size is >= 2.
     * 
     * The byte order does not apply when item size is 1 and also to the null dtype.
     * The orders can be: little endian or big endian.
     * Native order is only used for dtype string format.
     * 
     * @return npy_endianness the byte order of the dtype.
     */
    npy_endianness byte_order() const;
    /**
     * @brief The string represention of a dtype object according to NumPy dtype string format.
     * 
     * For a native type, the string is composed of 3 or 4 characters.
     * The first character is the byte order: '<' for little endian and '>' for big endian.
     * The second character is the kind of the dtype: 'b' for booleans, 'i' for integers, 'u' for unsigned, 'f' for floating points, and 'c' for complex.
     * The last character or the last two characters are the string version of the item size which can take
     * 1 character for sizes 1, 2, 4, 8, and two characters for the size 16.
     * 
     * @return std::string the string representation of a dtype object.
     */
    std::string str() const;

    /**
     * @brief Return an implemented dtype given the type provided by the user.
     * 
     * The only implemented and accepted C++ types are:
     * bool for 8-bit boolean value, 
     * char or int8_t for 8-bit signed integer value, 
     * int16_t or short for 16-bit signed integer value,
     * int32_t or int for 32-bit signed integer value,
     * int64_t or long for 64-bit signed integer value,
     * uint8_t or unsigned char for 8-bit unsigned integer value,
     * uint16_t or unsigned short for 16-bit unsigned integer value,
     * uint32_t or unsigned int for 32-bit unsigned integer value,
     * uint64_t or unsigned long for 64-bit unsigned integer value,
     * float for 32-bit floating point value,
     * double for 64-bit floating point value,
     * long double for 128-bit floating point value,
     * std::complex<float> for 64-bit complex,
     * std::complex<double> for 128-bit complex,
     * std::complex<long double> for 256-bit complex,
     * 
     * If the user provided type is not one of the above listed, then it is returned the null dtype.
     * 
     * @tparam T The C++ type the user want to describe as ndtype.
     * @return npy_dtype The corresponding dtype of null dtype if the C++ type is not implemented or accepted.
     */
    template<typename T> static npy_dtype from_type() noexcept
    {
        // Check if the user provided type is one of the implemented ones.
        // If yes, then return the dtype, otherwise return the null dtype.
        if(std::is_same<T, bool>::value) return npy_dtype::bool_8();
        else if(std::is_same<T, int8_t>::value || std::is_same<T, char>::value) return npy_dtype::int_8();
        else if(std::is_same<T, int16_t>::value) return npy_dtype::int_16();
        else if(std::is_same<T, int32_t>::value) return npy_dtype::int_32();
        else if(std::is_same<T, int64_t>::value) return npy_dtype::int_64();
        else if(std::is_same<T, uint8_t>::value) return npy_dtype::uint_8();
        else if(std::is_same<T, uint16_t>::value) return npy_dtype::uint_16();
        else if(std::is_same<T, uint32_t>::value) return npy_dtype::uint_32();
        else if(std::is_same<T, uint64_t>::value) return npy_dtype::uint_64();
        else if(std::is_same<T, float>::value) return npy_dtype::float_32();
        else if(std::is_same<T, double>::value) return npy_dtype::float_64();
        else if(std::is_same<T, long double>::value) return npy_dtype::float_128();
        else if(std::is_same<T, std::complex<float>>::value) return npy_dtype::complex_64();
        else if(std::is_same<T, std::complex<double>>::value) return npy_dtype::complex_128();
        else if(std::is_same<T, std::complex<long double>>::value) return npy_dtype::complex_256();
        else return npy_dtype::null();
    }
    /**
     * @brief Given a dtype string representation, return the correspoding dtype object.
     * 
     * For a native type, the string is composed of 3 or 4 characters.
     * The first character is the byte order: '<' for little endian and '>' for big endian, '=' for the machine native endianess, and '|' for the single byte types, like char or boolean.
     * If the endianess character is omitted, then the endianess is assumed to be equal to the machine one.
     * The second character is the kind of the dtype: 'b' for booleans, 'i' for integers, 'u' for unsigned, 'f' for floating points, and 'c' for complex.
     * This character is mandatory.
     * The last character or the last two characters are the string version of the item size which can take
     * 1 character for sizes 1, 2, 4, 8, and two characters for the size 16.
     * Notes that not all the possible combinations are possible and implemented.
     * 
     * Boolean
     * The only allowed combination is "|b1".
     * 
     * Integers
     * The only allowed combination for item size 1 is "|i1".
     * The only allowed combinations for item sizes >= 2 are "[<>=]?i[2,4,8]?"
     * If both endianess and item size are omitted, then it is assumed to be int64_t.
     * 
     * Unsigned Integers
     * The only allowed combination for item size 1 is "|u1".
     * The only allowed combinations for item sizes >= 2 are "[<>=]?u[2,4,8]?"
     * If both endianess and item size are omitted, then it is assumed to be uint64_t.
     * 
     * Floating Points
     * The only allowed combinations are "[<>=]?f[4,8,16]?"
     * If both endianess and item size are omitted, then it is assumed to be double.
     * 
     * Complex
     * The only allowed combinations are "[<>=]?c[8,16,32]?"
     * If both endianess and item size are omitted, then it is assumed to be std::complex<double>.
     * 
     * If the string is malformed or the type requested is not implemented, then it is returned the null dtype.
     * 
     * @param dtype_string the dtype string representation.
     * @return npy_dtype the dtype object for the dtype string, null if the string is wrong.
     */
    static npy_dtype from_string(const std::string& dtype_string) noexcept;

    /**
     * @brief Return a 8-bit boolean value, equivalent to a bool C++ type.
     * 
     * @return npy_dtype boolean dtype.
     */
    static npy_dtype bool_8() noexcept;
    /**
     * @brief Return a 8-bit signed integer value, equivalent to a char and int8_t C++ type.
     * 
     * @return npy_dtype 8-bit int dtype.
     */
    static npy_dtype int_8() noexcept;
    /**
     * @brief Return a 16-bit signed integer value, equivalent to a shprt and int16_t C++ type.
     * 
     * @return npy_dtype 16-bit int dtype.
     */
    static npy_dtype int_16() noexcept;
    /**
     * @brief Return a 32-bit signed integer value, equivalent to a int and int32_t C++ type.
     * 
     * @return npy_dtype 32-bit int dtype.
     */
    static npy_dtype int_32() noexcept;
    /**
     * @brief Return a 64-bit signed integer value, equivalent to a long and int64_t C++ type.
     * 
     * @return npy_dtype 64-bit int dtype.
     */
    static npy_dtype int_64() noexcept;
    /**
     * @brief Return a 8-bit unsigned integer value, equivalent to a unsigned char and uint8_t C++ type.
     * 
     * @return npy_dtype 8-bit unsigned int dtype.
     */
    static npy_dtype uint_8() noexcept;
    /**
     * @brief Return a 16-bit unsigned integer value, equivalent to a unsigned short and uint16_t C++ type.
     * 
     * @return npy_dtype 16-bit unsigned int dtype.
     */
    static npy_dtype uint_16() noexcept;
    /**
     * @brief Return a 32-bit unsigned integer value, equivalent to a unsigned int and uint32_t C++ type.
     * 
     * @return npy_dtype 32-bit unsigned int dtype.
     */
    static npy_dtype uint_32() noexcept;
    /**
     * @brief Return a 64-bit unsigned integer value, equivalent to a unsigned long and uint64_t C++ type.
     * 
     * @return npy_dtype 64-bit unsigned int dtype.
     */
    static npy_dtype uint_64() noexcept;
    /**
     * @brief Return a 32-bit floating value, equivalent to a float C++ type.
     * 
     * @return npy_dtype 32-bit floating value dtype.
     */
    static npy_dtype float_32() noexcept;
    /**
     * @brief Return a 64-bit floating value, equivalent to a double C++ type.
     * 
     * @return npy_dtype 64-bit floating value dtype.
     */
    static npy_dtype float_64() noexcept;
    /**
     * @brief Return a 128-bit floating value, equivalent to a long double C++ type.
     * 
     * @return npy_dtype 128-bit floating value dtype.
     */
    static npy_dtype float_128() noexcept;
    /**
     * @brief Return a 64-bit complex, equivalent to a std::complex<float> C++ type.
     * 
     * @return npy_dtype 64-bit complex dtype.
     */
    static npy_dtype complex_64() noexcept;
    /**
     * @brief Return a 128-bit complex, equivalent to a std::complex<double> C++ type.
     * 
     * @return npy_dtype 128-bit complex dtype.
     */
    static npy_dtype complex_128() noexcept;
    /**
     * @brief Return a 256-bit complex, equivalent to a std::complex<long double> C++ type.
     * 
     * @return npy_dtype 256-bit complex dtype.
     */
    static npy_dtype complex_256() noexcept;    
    /**
     * @brief Return a null dtype.
     * 
     * The null dtype object has the following invariants:
     * kind = npy_dtype_kind::unknown
     * item size = 0
     * byte order = npy_endianess::not_applicable.
     * 
     * @return npy_dtype null dtype.
     */
    static npy_dtype null() noexcept;
    
private:
    /**
     * @brief Private constructor that construct a new dtype with the user provided inputs.
     * 
     * It is used by the static methods to create the implemeted dtypes.
     * 
     * @param kind dtype's kind.
     * @param item_size dtype size in bytes.
     * @param byte_order the endianess of the dtype.
     */
    npy_dtype(npy_dtype_kind kind, size_t item_size, npy_endianness byte_order=get_endianess()) noexcept;

    npy_dtype_kind _kind; // The private dtype's kind.
    size_t _item_size; // The private item size in bytes.
    npy_endianness _byte_order; // The private dtype's byte order.
};



#endif /* D4AB99B8_10DA_46D7_9CFA_559B460F6ABA */
