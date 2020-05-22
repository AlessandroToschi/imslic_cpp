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

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}