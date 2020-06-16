#include <iostream>
#include <future>
#include <cassert>

#include "npy_array/npy_array.h"

template<typename T>
inline T sub2ind(const T x, const T y, const T w)
{
    return y * w + x;
}

template<typename T>
inline std::pair<T, T> ind2sub(const T index, const T w)
{
    return std::make_pair(index / w, index % w);
}

template<typename T>
inline bool check_boundaries(const T inclusive_min, const T exclusive_max, const T value)
{
    return inclusive_min <= value && value < exclusive_max;
}

void parallel_rgb_to_lab(const uint8_t* start, const uint8_t* end, float* out)
{
    float scaled_rgb_pixel[3];
    float xyz_pixel[3];
    float fxfyfz_pixel[3];

    for(auto i = start; i != end; i += 3, out += 3)
    {
        scaled_rgb_pixel[0] = float(*i) * 0.003921569f;
        scaled_rgb_pixel[1] = float(*(i + 1)) * 0.003921569f;
        scaled_rgb_pixel[2] = float(*(i + 2)) * 0.003921569f;

        scaled_rgb_pixel[0] = scaled_rgb_pixel[0] <= 0.04045f ? scaled_rgb_pixel[0] * 0.077399381f : powf((scaled_rgb_pixel[0] + 0.055f) * 0.947867299f, 2.4f); 
        scaled_rgb_pixel[1] = scaled_rgb_pixel[1] <= 0.04045f ? scaled_rgb_pixel[1] * 0.077399381f : powf((scaled_rgb_pixel[1] + 0.055f) * 0.947867299f, 2.4f); 
        scaled_rgb_pixel[2] = scaled_rgb_pixel[2] <= 0.04045f ? scaled_rgb_pixel[2] * 0.077399381f : powf((scaled_rgb_pixel[2] + 0.055f) * 0.947867299f, 2.4f); 

        xyz_pixel[0] = (0.412453f * scaled_rgb_pixel[0] + 0.35758f * scaled_rgb_pixel[1] + 0.180423f * scaled_rgb_pixel[2]) / 0.950456f;
        xyz_pixel[1] = 0.212671f * scaled_rgb_pixel[0] + 0.71516f * scaled_rgb_pixel[1] + 0.072169f * scaled_rgb_pixel[2];
        xyz_pixel[2] = (0.019334f * scaled_rgb_pixel[0] + 0.119193f * scaled_rgb_pixel[1] + 0.950227f * scaled_rgb_pixel[2]) / 1.088754f;

        fxfyfz_pixel[0] = xyz_pixel[0] > 0.008856f ? std::cbrtf(xyz_pixel[0]) : 7.787f * xyz_pixel[0] + 0.137931034f;
        fxfyfz_pixel[1] = xyz_pixel[1] > 0.008856f ? std::cbrtf(xyz_pixel[1]) : 7.787f * xyz_pixel[1] + 0.137931034f;
        fxfyfz_pixel[2] = xyz_pixel[2] > 0.008856f ? std::cbrtf(xyz_pixel[2]) : 7.787f * xyz_pixel[2] + 0.137931034f;
        
        *out = xyz_pixel[1] > 0.008856f ? 116.0f * fxfyfz_pixel[1] - 16.0f : 903.3f * fxfyfz_pixel[1];
        *(out + 1) = 500.0f * (fxfyfz_pixel[0] - fxfyfz_pixel[1]);
        *(out + 2) = 200.0f * (fxfyfz_pixel[1] - fxfyfz_pixel[2]);
    }
}

npy_array<float> rgb_to_lab(const npy_array<uint8_t>& rgb_image)
{
    npy_array<float> lab_image{rgb_image.shape()};

    std::vector<std::future<void>> futures{};
    futures.reserve(rgb_image.shape()[0]);

    const auto stride = rgb_image.shape()[1] * rgb_image.shape()[2];
    auto rgb_start = rgb_image.begin();
    auto rgb_end = rgb_start + stride;
    auto lab_start = lab_image.begin();

    for(auto y = 0UL; y != rgb_image.shape()[0]; y++)
    {
        futures.push_back(std::async(std::launch::async | std::launch::deferred, parallel_rgb_to_lab, rgb_start, rgb_end, lab_start));
        rgb_start = rgb_end;
        rgb_end += stride;
        lab_start += stride;
    }

    for(auto& future : futures)
    {
        future.get();
    }

    return std::move(lab_image);
}

void parallel_area(const npy_array<float>& lab_image, const size_t x_start, const size_t x_end, const size_t y_start, const size_t y_end)
{
    const auto width = x_end - x_start;
    const auto height = y_end - y_start;

    /*
    const std::array<std::pair<long, long>, 4> aaa = {
        std::make_pair<0L, 0L>, std::make_pair<1L, 1L>, std::make_pair<2L, 2L>, std::make_pair<3L, 3L>
    };
    */
    //float sub_lab_image[(width + 2) * (height + 2) * lab_image.shape()[2]];
    /*
    const std::array<std::array<std::pair<long, long>, 4>, 4> offsets = {
        {{0L, -1L}, {-1L, -1L}, {-1L, 0L}, {0L, 0L}},
        {{0L, 0L}, {-1L, 0L}, {-1L, 1L}, {0L, 1L}},
        {{0L, 0L}, {0L, 1L}, {1L, 1L}, {1L, 0L}},
        {{0L, 0L}, {1L, 0L}, {1L, -1L}, {0L, -1L}}
    };
    */
    for(auto y = y_start; y != y_end; y++)
    {
        for(auto x = x_start; x != x_end; x++)
        {
            float neighbors[27];

            for(auto dy = -1L; dy <= 1L; dy++)
            {
                for(auto dx = -1L; dx <= 1L; dx++)
                {
                    const long neighbor_x = long(x) + dx;
                    const long neighbor_y = long(y) + dy;
                    const auto between_boundaries = check_boundaries(0L, lab_image.shape()[1], neighbor_x) && check_boundaries(0L, lab_image.shape()[0], neighbor_y);
                    //const auto index = sub2ind(1L + dx, 1L + dx, 9L)
                    //neighbors[index] = check_boundaries(0L, x + dx, lab_image.shape()[1]) && check_boundaries(0L, y + dy, lab_image.shape()[0]) ? 
                }
            }
        }
    }
}

npy_array<float> compute_area(const npy_array<float>& lab_image)
{
    const size_t block_size = 10;
    npy_array<float> area{{lab_image.shape()[0], lab_image.shape()[1]}};

    const auto block_y = (lab_image.shape()[0] + block_size - 1) / block_size;
    const auto block_x = (lab_image.shape()[1] + block_size - 1) / block_size;

    std::vector<std::future<void>> futures{};
    futures.reserve(block_x * block_y);

    for(auto y = 0UL; y != block_y; y++)
    {
        const auto y_start = y * block_size;
        const auto y_end = std::min(lab_image.shape()[0], (y + 1) * block_size);

        for(auto x = 0UL; x != block_x; x++)
        {
            const auto x_start = x * block_size;
            const auto x_end = std::min(lab_image.shape()[1], (x + 1) * block_size);

            futures.push_back(std::async(
                std::launch::async | std::launch::deferred,
                parallel_area,
                std::ref(lab_image), x_start, x_end, y_start, y_end
            ));


            //std::cout << "(" << y_start << ", " << y_end << ") - (" << x_start << ", " << x_end << ")" << std::endl;
            //futures.push_back(std::async(std::launch::async | std::launch::deferred, ))

        }
    }

    for(auto& future : futures)
    {
        future.get();
    }

    return std::move(area);
}

int main(int argc, char* argv[])
{
    npy_array<uint8_t> rgb_image{"./image.npy"};
    npy_array<float> lab_image = rgb_to_lab(rgb_image);
    npy_array<float> area = compute_area(lab_image);

    return EXIT_SUCCESS;
}