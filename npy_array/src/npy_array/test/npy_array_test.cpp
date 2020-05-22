#define DEBUG 0

#include "npy_array/npy_array.h"
#include <gtest/gtest.h>



TEST(NPYArrayTest, ConstructorPathStringTest)
{
    try
    {
        npy_array<float>{""};
        FAIL();
    }
    catch(const npy_array_exception& npy_exception)
    {
        EXPECT_EQ(npy_exception.exception_type(), npy_array_exception_type::input_output_error);
    }

    try
    {
        npy_array<float>{"/test/ereregfdgdfgdgdfgfdgfgdgfdfggdf.npy"};
        FAIL();
    }
    catch(const npy_array_exception& npy_exception)
    {
        EXPECT_EQ(npy_exception.exception_type(), npy_array_exception_type::input_output_error);
    }
}    

TEST(NPYArrayTest, ConstructorMagicStringTest)
{
    try
    {
        npy_array<float>{"./test_resources/invalid_magic_string.npy"};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::invalid_magic_string);
    }

    try
    {
        npy_array<float>{"./test_resources/carbon.png"};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::invalid_magic_string);
    }

    try
    {
        npy_array<float>{"./test_resources/archive.npy"};
        SUCCEED();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_NE(e.exception_type(), npy_array_exception_type::invalid_magic_string);
    }
}

TEST(NPYArrayTest, ConstructorMajorVersion)
{
    try
    {
        npy_array<float>{"./test_resources/version_2.npy"};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::unsupported_version);
    }

    try
    {
        npy_array<float>{"./test_resources/version_3.npy"};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::unsupported_version);
    }

    try
    {
        npy_array<float>{"./test_resources/version_0.npy"};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::unsupported_version);
    }

    try
    {
        npy_array<float>{"./test_resources/archive.npy"};
        SUCCEED();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_NE(e.exception_type(), npy_array_exception_type::unsupported_version);
    }
}

TEST(NPYArrayTest, ConstructorHeaderParsing)
{
    try
    {
        npy_array<float>{"./test_resources/null_header.npy"};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::ill_formed_header);
    }

    try
    {
        npy_array<float>{"./test_resources/empty_header.npy"};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::ill_formed_header);
    }

    try
    {
        npy_array<float>{"./test_resources/fake_dtype.npy"};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::ill_formed_header);
    }

    try
    {
        npy_array<float>{"./test_resources/fake_descr.npy"};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::ill_formed_header);
    }

    try
    {
        npy_array<float>{"./test_resources/fake_fortran_order.npy"};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::ill_formed_header);
    }

    try
    {
        npy_array<float>{"./test_resources/fake_fortran_value.npy"};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::ill_formed_header);
    }

    try
    {
        npy_array<float>{"./test_resources/fake_shape.npy"};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::ill_formed_header);
    }

    try
    {
        npy_array<float>{"./test_resources/fake_shape_2.npy"};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::ill_formed_header);
    }

    try
    {
        npy_array<float>{"./test_resources/fake_shape_3.npy"};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::ill_formed_header);
    }

    EXPECT_NO_THROW(npy_array<long>{"./test_resources/fake_shape_4.npy"});
}

TEST(NPYArrayTest, StringConstructor)
{   
    EXPECT_NO_THROW(npy_array<float>{"./test_resources/archive.npy"});
    npy_array<float> array{"./test_resources/archive.npy"};
    std::vector<size_t> shape{1, 256, 13, 13};
    EXPECT_EQ(array.byte_size(), sizeof(float) * 256 * 13 * 13);
    EXPECT_EQ(array.dtype(), npy_dtype::float_32());
    EXPECT_EQ(array.fortran_order(), false);
    EXPECT_EQ(array.shape(), shape);
}

TEST(NPYArrayTest, CopyShapeConstructor)
{   
    std::vector<size_t> shape{1, 256, 13, 13};
    npy_array<float> array{shape};

    EXPECT_EQ(array.byte_size(), sizeof(float) * 256 * 13 * 13);
    EXPECT_EQ(array.dtype(), npy_dtype::float_32());
    EXPECT_EQ(array.fortran_order(), false);
    EXPECT_EQ(array.shape(), shape);

    EXPECT_EQ(shape.size(), 4);

    EXPECT_THROW(npy_array<std::bitset<3>>{shape}, npy_array_exception);
}

TEST(NPYArrayTest, MoveShapeConstructor)
{   
    std::vector<size_t> shape{1, 256, 13, 13};
    npy_array<float> array{std::move(shape)};

    EXPECT_EQ(array.byte_size(), sizeof(float) * 256 * 13 * 13);
    EXPECT_EQ(array.dtype(), npy_dtype::float_32());
    EXPECT_EQ(array.fortran_order(), false);
    EXPECT_NE(array.shape(), shape);

    EXPECT_EQ(shape.size(), 0);

    EXPECT_THROW(npy_array<std::bitset<3>>{shape}, npy_array_exception);
}

TEST(NPYArrayTest, ShapeListConstructor)
{   
    npy_array<float> array{{1, 256, 13, 13}};

    EXPECT_EQ(array.byte_size(), sizeof(float) * 256 * 13 * 13);
    EXPECT_EQ(array.dtype(), npy_dtype::float_32());
    EXPECT_EQ(array.fortran_order(), false);
}

TEST(NPYArrayTest, ShapeDataCopyConstructor)
{
    std::vector<size_t> shape{{2, 3}};
    std::vector<size_t> wrong_shape{{2, 3, 4}};
    std::vector<uint8_t> data{{1, 2, 3, 4, 5, 6}};
    std::vector<uint8_t> wrong_data{{1, 2, 3, 4}};

    npy_array<uint8_t> arr{shape, data};

    EXPECT_EQ(arr.byte_size(), sizeof(uint8_t) * 2 * 3);
    EXPECT_EQ(arr.dtype(), npy_dtype::uint_8());
    EXPECT_EQ(arr.fortran_order(), false);
    EXPECT_EQ(arr.shape(), shape);
    EXPECT_EQ(arr.size(), 2 * 3);

    try
    {
        npy_array<uint8_t>{wrong_shape, data};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::unmatched_shape_data);
    }

    try
    {
        npy_array<uint8_t>{shape, wrong_data};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::unmatched_shape_data);
    }
    
    try
    {
        npy_array<uint8_t>{wrong_shape, wrong_data};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::unmatched_shape_data);
    }
}

TEST(NPYArrayTest, ShapeDataMoveConstructor)
{
    std::vector<size_t> shape{{2, 3}};
    std::vector<size_t> wrong_shape{{2, 3, 4}};
    std::vector<uint8_t> data{{1, 2, 3, 4, 5, 6}};
    std::vector<uint8_t> wrong_data{{1, 2, 3, 4}};

    npy_array<uint8_t> arr{std::move(shape), std::move(data)};

    EXPECT_EQ(arr.byte_size(), sizeof(uint8_t) * 2 * 3);
    EXPECT_EQ(arr.dtype(), npy_dtype::uint_8());
    EXPECT_EQ(arr.fortran_order(), false);
    EXPECT_EQ(arr.size(), 2 * 3);

    EXPECT_EQ(shape.size(), 0);
    EXPECT_EQ(data.size(), 0);

    try
    {
        std::vector<size_t> shape{{2, 3}};
        std::vector<size_t> wrong_shape{{2, 3, 4}};
        std::vector<uint8_t> data{{1, 2, 3, 4, 5, 6}};
        std::vector<uint8_t> wrong_data{{1, 2, 3, 4}};

        npy_array<uint8_t>{std::move(wrong_shape), std::move(data)};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::unmatched_shape_data);
    }

    try
    {
        std::vector<size_t> shape{{2, 3}};
        std::vector<size_t> wrong_shape{{2, 3, 4}};
        std::vector<uint8_t> data{{1, 2, 3, 4, 5, 6}};
        std::vector<uint8_t> wrong_data{{1, 2, 3, 4}};
        npy_array<uint8_t>{std::move(shape), std::move(wrong_data)};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::unmatched_shape_data);
    }
    
    try
    {
        std::vector<size_t> shape{{2, 3}};
        std::vector<size_t> wrong_shape{{2, 3, 4}};
        std::vector<uint8_t> data{{1, 2, 3, 4, 5, 6}};
        std::vector<uint8_t> wrong_data{{1, 2, 3, 4}};
        npy_array<uint8_t>{std::move(wrong_shape), std::move(wrong_data)};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::unmatched_shape_data);
    }
}

TEST(NPYArrayTest, ShapeDataListConstructor)
{
    npy_array<uint8_t> arr{{2, 3}, {1, 2, 3, 4, 5, 6}};

    EXPECT_EQ(arr.byte_size(), sizeof(uint8_t) * 2 * 3);
    EXPECT_EQ(arr.dtype(), npy_dtype::uint_8());
    EXPECT_EQ(arr.fortran_order(), false);
    EXPECT_EQ(arr.size(), 2 * 3);

    try
    {
        npy_array<uint8_t>{{2}, {1, 2, 3, 4, 5, 6}};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::unmatched_shape_data);
    }

    try
    {
        npy_array<uint8_t>{{2, 3}, {1, 3, 4, 5, 6}};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::unmatched_shape_data);
    }
    
    try
    {
        npy_array<uint8_t>{{2}, {1, 2, 3, 5, 6}};
        FAIL();
    }
    catch(const npy_array_exception& e)
    {
        EXPECT_EQ(e.exception_type(), npy_array_exception_type::unmatched_shape_data);
    }
}

TEST(NPYArrayTest, MoveTest)
{
    npy_array<float> p{"./test_resources/archive.npy"};
    npy_array<float> c{std::move(p)};
    std::vector<size_t> s{{1, 256, 13, 13}};

    EXPECT_EQ(p.byte_size(), 0);
    EXPECT_EQ(p.size(), 0);
    EXPECT_EQ(p.fortran_order(), false);
    EXPECT_EQ(p.shape(), std::vector<size_t>{});
    EXPECT_EQ(p.dtype(), npy_dtype::null());

    EXPECT_EQ(c.byte_size(), sizeof(float) * 256 * 13 * 13);
    EXPECT_EQ(c.size(), 256 * 13 * 13);
    EXPECT_EQ(c.fortran_order(), false);
    EXPECT_EQ(c.shape(), s);
    EXPECT_EQ(c.dtype(), npy_dtype::float_32());

    auto d = std::move(c);

    EXPECT_EQ(c.byte_size(), 0);
    EXPECT_EQ(c.size(), 0);
    EXPECT_EQ(c.fortran_order(), false);
    EXPECT_EQ(c.shape(), std::vector<size_t>{});
    EXPECT_EQ(c.dtype(), npy_dtype::null());

    EXPECT_EQ(d.byte_size(), sizeof(float) * 256 * 13 * 13);
    EXPECT_EQ(d.size(), 256 * 13 * 13);
    EXPECT_EQ(d.fortran_order(), false);
    EXPECT_EQ(d.shape(), s);
    EXPECT_EQ(d.dtype(), npy_dtype::float_32());
}

TEST(NPYArrayTest, CopyTest)
{
    npy_array<float> p{"./test_resources/archive.npy"};
    npy_array<float> c{p};
    std::vector<size_t> s{{1, 256, 13, 13}};
    
    EXPECT_EQ(p.byte_size(), sizeof(float) * 256 * 13 * 13);
    EXPECT_EQ(p.size(), 256 * 13 * 13);
    EXPECT_EQ(p.fortran_order(), false);
    EXPECT_EQ(p.shape(), s);
    EXPECT_EQ(p.dtype(), npy_dtype::float_32());

    EXPECT_EQ(c.byte_size(), sizeof(float) * 256 * 13 * 13);
    EXPECT_EQ(c.size(), 256 * 13 * 13);
    EXPECT_EQ(c.fortran_order(), false);
    EXPECT_EQ(c.shape(), s);
    EXPECT_EQ(c.dtype(), npy_dtype::float_32());

    EXPECT_EQ(c.byte_size(), p.byte_size());
    EXPECT_EQ(c.size(), p.size());
    EXPECT_EQ(c.fortran_order(), p.fortran_order());
    EXPECT_EQ(c.shape(), p.shape());
    EXPECT_EQ(c.dtype(), p.dtype());

    for(size_t i = 0; i < c.size(); i++)
    {
        EXPECT_FLOAT_EQ(c.data()[i], p.data()[i]);
        EXPECT_NE(&c.data()[i], &p.data()[i]);
    }

    npy_array<float> ss = p;

    EXPECT_EQ(p.byte_size(), sizeof(float) * 256 * 13 * 13);
    EXPECT_EQ(p.size(), 256 * 13 * 13);
    EXPECT_EQ(p.fortran_order(), false);
    EXPECT_EQ(p.shape(), s);
    EXPECT_EQ(p.dtype(), npy_dtype::float_32());

    EXPECT_EQ(ss.byte_size(), sizeof(float) * 256 * 13 * 13);
    EXPECT_EQ(ss.size(), 256 * 13 * 13);
    EXPECT_EQ(ss.fortran_order(), false);
    EXPECT_EQ(ss.shape(), s);
    EXPECT_EQ(ss.dtype(), npy_dtype::float_32());

    EXPECT_EQ(ss.byte_size(), p.byte_size());
    EXPECT_EQ(ss.size(), p.size());
    EXPECT_EQ(ss.fortran_order(), p.fortran_order());
    EXPECT_EQ(ss.shape(), p.shape());
    EXPECT_EQ(ss.dtype(), p.dtype());

    for(size_t i = 0; i < s.size(); i++)
    {
        EXPECT_FLOAT_EQ(ss.data()[i], p.data()[i]);
        EXPECT_NE(&ss.data()[i], &p.data()[i]);
    }
}

TEST(NPYArrayTest, DataTest)
{
    npy_array<long> range{"./test_resources/10.npy"};

    for(size_t i = 0; i != range.size(); i++)
    {
        EXPECT_EQ(range.data()[i], i);
    }

    for(size_t i = 0; i != range.size(); i++)
    {
        range.data()[i] = i + 10;
        EXPECT_EQ(range.data()[i], i + 10);
    }
}

TEST(NPYArrayTest, OperatorAtTest)
{
    npy_array<long> range{"./test_resources/10.npy"};

    for(size_t i = 0; i != range.size(); i++)
    {
        EXPECT_EQ(range[i], i);
    }

    for(size_t i = 0; i != range.size(); i++)
    {
        range[i] = i + 10;
        EXPECT_EQ(range[i], i + 10);
    }
}

TEST(NPYArrayTest, SaveTest)
{
    npy_array<float> a{"./test_resources/archive.npy"};

    for(size_t i = 0; i < a.size(); i++)
    {
        a[i] = 0.0f;
    }

    a.save("./test_resources/zero.npy");

    npy_array<uint8_t> b{{481, 321, 3}};
    b.save("sddf");
}


int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}