#include <iostream>
#include <math.h>

#include "npy_array/npy_array.h"

template<typename T_IN, typename T_OUT>
npy_array<T_OUT> convert_dtype(const npy_array<T_IN>& in_array)
{
    npy_array<T_OUT> out_array{in_array.shape()};

    for(size_t i = 0; i < in_array.size(); i++)
    {
        out_array[i] = T_OUT(in_array[i]);
    }

    return out_array;
}

npy_array<float> rgb_to_lab(const npy_array<float>& rgb_image)
{
    npy_array<float> lab_image{rgb_image.shape()};

    for(size_t h = 0; h < rgb_image.shape()[0]; h++)
    {
        for(size_t w = 0; w < rgb_image.shape()[1]; w++)
        {
            size_t index = h * (rgb_image.shape()[1] * rgb_image.shape()[2]) + w * rgb_image.shape()[2];

            const float* rgb_pixel = &(rgb_image.data()[index]);
            float* lab_pixel = &(lab_image.data()[index]);
            float scaled_rgb_pixel[3];
            float xyz_pixel[3];
            float fxfyfz_pixel[3];

            scaled_rgb_pixel[0] = rgb_pixel[0] * 0.003921569f;
            scaled_rgb_pixel[1] = rgb_pixel[1] * 0.003921569f;
            scaled_rgb_pixel[2] = rgb_pixel[2] * 0.003921569f;

            scaled_rgb_pixel[0] = scaled_rgb_pixel[0] <= 0.04045f ? scaled_rgb_pixel[0] * 0.077399381f : powf((scaled_rgb_pixel[0] + 0.055f) * 0.947867299f, 2.4f); 
            scaled_rgb_pixel[1] = scaled_rgb_pixel[1] <= 0.04045f ? scaled_rgb_pixel[1] * 0.077399381f : powf((scaled_rgb_pixel[1] + 0.055f) * 0.947867299f, 2.4f); 
            scaled_rgb_pixel[2] = scaled_rgb_pixel[2] <= 0.04045f ? scaled_rgb_pixel[2] * 0.077399381f : powf((scaled_rgb_pixel[2] + 0.055f) * 0.947867299f, 2.4f); 

            xyz_pixel[0] = (0.412453f * scaled_rgb_pixel[0] + 0.35758f * scaled_rgb_pixel[1] + 0.180423f * scaled_rgb_pixel[2]) / 0.950456f;
            xyz_pixel[1] = 0.212671f * scaled_rgb_pixel[0] + 0.71516f * scaled_rgb_pixel[1] + 0.072169f * scaled_rgb_pixel[2];
            xyz_pixel[2] = (0.019334f * scaled_rgb_pixel[0] + 0.119193f * scaled_rgb_pixel[1] + 0.950227f * scaled_rgb_pixel[2]) / 1.088754f;

            fxfyfz_pixel[0] = xyz_pixel[0] > 0.008856f ? std::cbrtf(xyz_pixel[0]) : 7.787f * xyz_pixel[0] + 0.137931034f;
            fxfyfz_pixel[1] = xyz_pixel[1] > 0.008856f ? std::cbrtf(xyz_pixel[1]) : 7.787f * xyz_pixel[1] + 0.137931034f;
            fxfyfz_pixel[2] = xyz_pixel[2] > 0.008856f ? std::cbrtf(xyz_pixel[2]) : 7.787f * xyz_pixel[2] + 0.137931034f;

            lab_pixel[0] = xyz_pixel[1] > 0.008856f ? 116.0f * fxfyfz_pixel[1] - 16.0f : 903.3f * fxfyfz_pixel[1];
            lab_pixel[1] = 500.0f * (fxfyfz_pixel[0] - fxfyfz_pixel[1]);
            lab_pixel[2] = 200.0f * (fxfyfz_pixel[1] - fxfyfz_pixel[2]);
        }
    }

    return lab_image;
}

template<typename T>
npy_array<T> pad(const npy_array<T>& array)
{
    npy_array<T> padded_array{{array.shape()[0] + 2, array.shape()[1] + 2, array.shape()[2]}};

    for(size_t h = 0; h < array.shape()[0]; h++)
    {
        size_t index = h * (array.shape()[1] * array.shape()[2]);
        size_t padded_index = (h + 1) * (padded_array.shape()[1] * padded_array.shape()[2]);

        if(h == 0)
        {
            std::memcpy(&padded_array.data()[padded_index + 1 - (padded_array.shape()[1] * padded_array.shape()[2])], &array.data()[index], sizeof(T) * (array.shape()[1] * array.shape()[2]));
        }
        else if(h == array.shape()[0] - 1)
        {
            std::memcpy(&padded_array.data()[padded_index + 1 + (padded_array.shape()[1] * padded_array.shape()[2])], &array.data()[index], sizeof(T) * (array.shape()[1] * array.shape()[2]));
        }

        std::memcpy(&padded_array.data()[padded_index + 1], &array.data()[index], sizeof(T) * (array.shape()[1] * array.shape()[2]));    

        padded_array[padded_index] = padded_array[padded_index + 1];
        padded_array[padded_index + (padded_array.shape()[1] * padded_array.shape()[2]) - 1] = padded_array[padded_index + (padded_array.shape()[1] * padded_array.shape()[2]) - 2];
    }

    return padded_array;
}


npy_array<float> compute_area(const npy_array<float>& lab_image)
{
    npy_array<float> area{{lab_image.shape()[0], lab_image.shape()[1]}};

    for(size_t h = 0; h < lab_image.shape()[0]; h++)
    {
        for(size_t w = 0; w < lab_image.shape()[1]; w++)
        {

        }
    }

    return area;
}

int main(int argc, char* argv[])
{
    const int region_size = 10;
    const int max_iterations = 15;

    npy_array<uint8_t> rgb_image{"./image.npy"};
    rgb_image.save("./my_results/rgb_image.npy");

    npy_array<float> float_rgb_image = std::move(convert_dtype<uint8_t, float>(rgb_image));
    float_rgb_image.save("./my_results/float_rgb_image.npy");

    //auto xyz_image = std::move(rgb_to_lab(float_rgb_image));
    //xyz_image.save("./my_results/xyz_image.npy");

    npy_array<float> lab_image = std::move(rgb_to_lab(float_rgb_image));
    lab_image.save("./my_results/lab_image.npy");

    {
        npy_array<float> padded_lab_image = std::move(pad(lab_image));
    }

    return EXIT_SUCCESS;
}