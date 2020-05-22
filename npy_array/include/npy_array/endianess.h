#ifndef FF2408FE_6542_4CBE_B7CB_1CC00ECE539D
#define FF2408FE_6542_4CBE_B7CB_1CC00ECE539D

#include <stdint.h>

enum npy_endianness
{
    little_endian = '<',
    big_endian = '>',
    native = '=',
    not_applicable = '|'
};

static union word_struct
{
    uint32_t integer_value;
    uint8_t byte_values[4];
} word = {0x10203040};

npy_endianness get_endianess()
{
    static npy_endianness machine_endianess = word.byte_values[0] == 0x10 ? npy_endianness::big_endian : npy_endianness::little_endian;
    return machine_endianess;
}

#endif /* FF2408FE_6542_4CBE_B7CB_1CC00ECE539D */