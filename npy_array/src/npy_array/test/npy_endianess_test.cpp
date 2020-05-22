#include <gtest/gtest.h>
#include <endian.h>

#include "npy_array/endianess.h"

TEST(NPYEndianessTest, MachineEndianessCorrectnessTest)
{
    npy_endianness machine_endianess = get_endianess();
    
    if(__BYTE_ORDER == __LITTLE_ENDIAN)
    {
        EXPECT_EQ(machine_endianess, npy_endianness::little_endian);
    }
    else if(__BYTE_ORDER == __BIG_ENDIAN)
    {
        EXPECT_EQ(machine_endianess, npy_endianness::big_endian);
    }
    else
    {
        FAIL();
    }
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}