#include <iostream>
#include <math.h>
#include <valarray>
#include <numeric>

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
    const size_t stride = array.shape()[1] * array.shape()[2];
    const size_t byte_stride = stride * sizeof(T);
    const size_t padded_stride = padded_array.shape()[1] * padded_array.shape()[2];

    for(size_t h = 0; h < array.shape()[0]; h++)
    {
        const size_t index = h * stride;
        const size_t padded_index = (h + 1) * padded_stride;

        if(h == 0)
        {
            std::memcpy(&padded_array.data()[padded_index + 3 - padded_stride], &array.data()[index], byte_stride);

            padded_array[padded_index - padded_stride] = padded_array[padded_index + 3 - padded_stride];
            padded_array[padded_index + 1 - padded_stride] = padded_array[padded_index + 4 - padded_stride];
            padded_array[padded_index + 2 - padded_stride] = padded_array[padded_index + 5 - padded_stride];

            padded_array[padded_index - 3] = padded_array[padded_index + padded_stride - 6 - padded_stride];
            padded_array[padded_index - 2] = padded_array[padded_index + padded_stride - 5 - padded_stride];
            padded_array[padded_index - 1] = padded_array[padded_index + padded_stride - 4 - padded_stride];
        }
        else if(h == array.shape()[0] - 1)
        {
            std::memcpy(&padded_array.data()[padded_index + 3 + padded_stride], &array.data()[index], byte_stride);

            padded_array[padded_index + padded_stride] = padded_array[padded_index + 3 + padded_stride];
            padded_array[padded_index + 1 + padded_stride] = padded_array[padded_index + 4 + padded_stride];
            padded_array[padded_index + 2 + padded_stride] = padded_array[padded_index + 5 + padded_stride];

            padded_array[padded_index + 2 * padded_stride - 3] = padded_array[padded_index + padded_stride - 6 + padded_stride];
            padded_array[padded_index + 2 * padded_stride - 2] = padded_array[padded_index + padded_stride - 5 + padded_stride];
            padded_array[padded_index + 2 * padded_stride - 1] = padded_array[padded_index + padded_stride - 4 + padded_stride];
        }

        std::memcpy(&padded_array.data()[padded_index + 3], &array.data()[index], byte_stride);    

        padded_array[padded_index] = padded_array[padded_index + 3];
        padded_array[padded_index + 1] = padded_array[padded_index + 4];
        padded_array[padded_index + 2] = padded_array[padded_index + 5];

        padded_array[padded_index + padded_stride - 3] = padded_array[padded_index + padded_stride - 6];
        padded_array[padded_index + padded_stride - 2] = padded_array[padded_index + padded_stride - 5];
        padded_array[padded_index + padded_stride - 1] = padded_array[padded_index + padded_stride - 4];
    }

    return padded_array;
}

std::valarray<float> get_corner(const float* lab_pixel_ptr, const std::vector<long>& offsets, const float x, const float y)
{
    std::valarray<float> corner{{0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

    for(int offset : offsets)
    {
        corner[2] += *(lab_pixel_ptr + offset);
        corner[3] += *(lab_pixel_ptr + offset + 1);
        corner[4] += *(lab_pixel_ptr + offset + 2);
    }

    corner[0] = float(x);
    corner[1] = float(y);

    corner *= 0.25f;

    return corner;
}

float delta(const std::valarray<float>& x, const std::valarray<float>& y)
{
    auto dot_product = (x * y).sum();
    auto x_norm = std::sqrt(std::pow(x, 2.0f).sum());
    auto y_norm = std::sqrt(std::pow(y, 2.0f).sum());
    auto angle = std::sqrt(1.0f - std::pow(dot_product / (x_norm * y_norm), 2.0f));
    return 0.5f * x_norm * y_norm * angle;
}


npy_array<float> compute_area(const npy_array<float>& padded_lab_image, const std::vector<size_t>& shape)
{
    npy_array<float> area{{shape[0], shape[1]}};
    const size_t padded_stride = padded_lab_image.shape()[1] * padded_lab_image.shape()[2];
    const std::vector<long> offsets{{
        -long(padded_stride) - 3,
        -long(padded_stride),
        -long(padded_stride) + 3,
        -3,
        0,
        3,
        long(padded_stride) - 3,
        long(padded_stride),
        long(padded_stride) + 3
    }};
    const std::vector<long> north_west_offsets{{offsets[0], offsets[1], offsets[3], offsets[4]}};
    const std::vector<long> north_east_offsets{{offsets[1], offsets[2], offsets[4], offsets[5]}};
    const std::vector<long> south_west_offsets{{offsets[3], offsets[4], offsets[6], offsets[7]}};
    const std::vector<long> south_east_offsets{{offsets[4], offsets[5], offsets[7], offsets[8]}};

    for(size_t h = 0; h < area.shape()[0]; h++)
    {
        for(size_t w = 0; w < area.shape()[1]; w++)
        {
            size_t padded_index = (h + 1) * padded_stride + (3 * (w + 1));
            const float* lab_pixel_ptr = &padded_lab_image.data()[padded_index];

            auto a1 = get_corner(lab_pixel_ptr, north_west_offsets, 4.0f * float(w) - 2.0f, 4.0f * float(h) - 2.0f);
            auto a2 = get_corner(lab_pixel_ptr, south_west_offsets, 4.0f * float(w) - 2.0f, 4.0f * float(h) + 2.0f);
            auto a3 = get_corner(lab_pixel_ptr, south_east_offsets, 4.0f * float(w) + 2.0f, 4.0f * float(h) + 2.0f);
            auto a4 = get_corner(lab_pixel_ptr, north_east_offsets, 4.0f * float(w) + 2.0f, 4.0f * float(h) - 2.0f);

            auto a21 = a2 - a1;
            auto a23 = a2 - a3;
            auto a43 = a4 - a3;
            auto a41 = a4 - a1;

            area[h * area.shape()[1] + w] = delta(a21, a23) + delta(a43, a41);
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

    npy_array<float> lab_image = std::move(rgb_to_lab(float_rgb_image));
    lab_image.save("./my_results/lab_image.npy");

    npy_array<float> padded_lab_image = std::move(pad(lab_image));
    padded_lab_image.save("./my_results/padded_lab_image.npy");

    npy_array<float> area = std::move(compute_area(padded_lab_image, lab_image.shape()));
    area.save("./my_results/area.npy");

    npy_array<float> cumulative_area{area.shape()};
    std::partial_sum(area.data(), area.data() + area.size(), cumulative_area.data(), std::plus<float>());
    cumulative_area.save("./my_results/cumulative_area.npy");

    return EXIT_SUCCESS;
}