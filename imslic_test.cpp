#include <gtest/gtest.h>
#include "npy_array/npy_array.h"

TEST(IMSLICTest, RGBImage)
{
    npy_array<uint8_t> cv_rgb_image{"./cv_results/rgb_image.npy"};
    npy_array<uint8_t> my_rgb_image{"./my_results/rgb_image.npy"};

    EXPECT_EQ(cv_rgb_image.byte_size(), my_rgb_image.byte_size());
    EXPECT_EQ(cv_rgb_image.size(), my_rgb_image.size());
    EXPECT_EQ(cv_rgb_image.shape(), my_rgb_image.shape());

    for(size_t i = 0; i < cv_rgb_image.size(); i++)
    {
        EXPECT_EQ(cv_rgb_image[i], my_rgb_image[i]);
    }
}

TEST(IMSLICTest, FloatRGBImage)
{
    npy_array<float> cv_rgb_image{"./cv_results/float_rgb_image.npy"};
    npy_array<float> my_rgb_image{"./my_results/float_rgb_image.npy"};

    EXPECT_EQ(cv_rgb_image.byte_size(), my_rgb_image.byte_size());
    EXPECT_EQ(cv_rgb_image.size(), my_rgb_image.size());
    EXPECT_EQ(cv_rgb_image.shape(), my_rgb_image.shape());

    for(size_t i = 0; i < cv_rgb_image.size(); i++)
    {
        EXPECT_NEAR(cv_rgb_image[i], my_rgb_image[i], 1E-5f);
    }
}

TEST(IMSLICTest, LabImage)
{
    npy_array<float> cv_rgb_image{"./cv_results/lab_image.npy"};
    npy_array<float> my_rgb_image{"./my_results/lab_image.npy"};

    EXPECT_EQ(cv_rgb_image.byte_size(), my_rgb_image.byte_size());
    EXPECT_EQ(cv_rgb_image.size(), my_rgb_image.size());
    EXPECT_EQ(cv_rgb_image.shape(), my_rgb_image.shape());

    for(size_t i = 0; i < cv_rgb_image.size(); i++)
    {
        EXPECT_NEAR(cv_rgb_image[i], my_rgb_image[i], 0.5f);
    }
}

TEST(IMSLICTest, PaddedLabImage)
{
    npy_array<float> cv_rgb_image{"./cv_results/padded_lab_image.npy"};
    npy_array<float> my_rgb_image{"./my_results/padded_lab_image.npy"};

    EXPECT_EQ(cv_rgb_image.byte_size(), my_rgb_image.byte_size());
    EXPECT_EQ(cv_rgb_image.size(), my_rgb_image.size());
    EXPECT_EQ(cv_rgb_image.shape(), my_rgb_image.shape());

    for(size_t i = 0; i < cv_rgb_image.size(); i++)
    {
        EXPECT_FLOAT_EQ(cv_rgb_image[i], my_rgb_image[i]);
    }
}

TEST(IMSLICTest, Area)
{
    npy_array<float> cv_rgb_image{"./cv_results/area.npy"};
    npy_array<float> my_rgb_image{"./my_results/area.npy"};

    EXPECT_EQ(cv_rgb_image.byte_size(), my_rgb_image.byte_size());
    EXPECT_EQ(cv_rgb_image.size(), my_rgb_image.size());
    EXPECT_EQ(cv_rgb_image.shape(), my_rgb_image.shape());

    for(size_t i = 0; i < cv_rgb_image.size(); i++)
    {
        EXPECT_NEAR(cv_rgb_image[i], my_rgb_image[i], 10E-3f);
    }
}

TEST(IMSLICTest, CumulativeArea)
{
    npy_array<float> cv_rgb_image{"./cv_results/cumulative_area.npy"};
    npy_array<float> my_rgb_image{"./my_results/cumulative_area.npy"};

    EXPECT_EQ(cv_rgb_image.byte_size(), my_rgb_image.byte_size());
    EXPECT_EQ(cv_rgb_image.size(), my_rgb_image.size());
    EXPECT_EQ(cv_rgb_image.shape(), my_rgb_image.shape());

    for(size_t i = 0; i < cv_rgb_image.size(); i++)
    {
        EXPECT_NEAR(cv_rgb_image[i], my_rgb_image[i], 10E-3f);

        if(i > 0)
        {
            EXPECT_GE(my_rgb_image[i], my_rgb_image[i - 1]);
            EXPECT_GE(cv_rgb_image[i], cv_rgb_image[i - 1]);
        }
    }
}

TEST(IMSLICTest, Seeds)
{
    npy_array<float> lab_image{"./my_results/lab_image.npy"};
    npy_array<float> seeds{"./my_results/seeds.npy"};

    for(size_t seed_index = 0; seed_index < seeds.shape()[0] - 1; seed_index++)
    {
        const size_t current_index = size_t(seeds[{seed_index, 0}]) * lab_image.shape()[1] + size_t(seeds[{seed_index, 1}]);
        const size_t next_index = size_t(seeds[{seed_index + 1, 0}]) * lab_image.shape()[1] + size_t(seeds[{seed_index + 1, 1}]);
        EXPECT_LT(current_index, next_index);
    }

    for(size_t seed_index = 0; seed_index != seeds.shape()[0]; seed_index++)
    {
        const size_t y = size_t(seeds[{seed_index, 0}]);
        const size_t x = size_t(seeds[{seed_index, 1}]);
        const float* seed_pixel = &seeds[{seed_index, 2}];
        const float* lab_pixel = &lab_image[{y, x, 0}];

        for(size_t i = 0; i != 3; i++)
        {
            EXPECT_EQ(seed_pixel[i], lab_pixel[i]);
        }
    }
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}