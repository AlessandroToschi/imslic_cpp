#include <iostream>
#include <future>
#include <cassert>
#include <valarray>
#include <chrono>

#include "npy_array/npy_array.h"

std::launch policy = std::launch::deferred | std::launch::async;

class timer
{
    typedef std::chrono::high_resolution_clock::time_point time_point;

public:
    timer() : _start{}, _end{} {}
    void start()
    {
        _start = std::chrono::high_resolution_clock::now();
    }

    void stop_and_print()
    {
        _end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start);
        
        std::cout << "Duration: " << duration.count() << " ms" << std::endl;
    }

private:
    time_point _start;
    time_point _end;
};

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
        futures.push_back(std::async(policy, parallel_rgb_to_lab, rgb_start, rgb_end, lab_start));
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

float delta(const std::valarray<float>& x, const std::valarray<float>& y)
{
    auto dot_product = (x * y).sum();
    auto x_norm = std::sqrt(std::pow(x, 2.0f).sum());
    auto y_norm = std::sqrt(std::pow(y, 2.0f).sum());
    auto angle = std::sqrt(1.0f - std::pow(dot_product / (x_norm * y_norm), 2.0f));
    return 0.5f * x_norm * y_norm * angle;
}

void parallel_area(const float* lab_start, const float* lab_end, const float* previous, const float* next, float* area_start)
{
    std::valarray<float> a1(0.0f, 5), a2(0.0f, 5), a3(0.0f, 5), a4(0.0f, 5);

    for(auto x = lab_start, p = previous, n = next; x != lab_end; x +=3, p += 3, n += 3, area_start += 1)
    {
        a1[2] = x[0];
        a1[3] = x[1];
        a1[4] = x[2];

        a2 = a1;
        a3 = a1;
        a4 = a1;

        a1[2] += x == lab_start ? previous[0] : *(previous - 3);
        a1[3] += x == lab_start ? previous[1] : *(previous - 2);
        a1[4] += x == lab_start ? previous[2] : *(previous - 1);

        a1[2] += previous[0];
        a1[3] += previous[1];
        a1[4] += previous[2];

        a4[2] += previous[0];
        a4[3] += previous[1];
        a4[4] += previous[2];

        a4[2] += x == lab_end - 3 ? previous[0] : previous[3];
        a4[3] += x == lab_end - 3 ? previous[1] : previous[4];
        a4[4] += x == lab_end - 3 ? previous[2] : previous[5];

        a1[2] += x == lab_start ? x[0] : *(x - 3); 
        a1[3] += x == lab_start ? x[1] : *(x - 2); 
        a1[4] += x == lab_start ? x[2] : *(x - 1); 

        a2[2] += x == lab_start ? x[0] : *(x - 3); 
        a2[3] += x == lab_start ? x[1] : *(x - 2); 
        a2[4] += x == lab_start ? x[2] : *(x - 1); 

        a4[2] += x == lab_end - 3 ? x[0] : *(x + 3);
        a4[3] += x == lab_end - 3 ? x[1] : *(x + 2);
        a4[4] += x == lab_end - 3 ? x[2] : *(x + 1);

        a3[2] += x == lab_end - 3 ? x[0] : *(x + 3);
        a3[3] += x == lab_end - 3 ? x[1] : *(x + 2);
        a3[4] += x == lab_end - 3 ? x[2] : *(x + 1);

        a2[2] += x == lab_start ? next[0] : *(next - 3);
        a2[3] += x == lab_start ? next[1] : *(next - 2);
        a2[4] += x == lab_start ? next[2] : *(next - 1);

        a2[2] += next[0];
        a2[3] += next[1];
        a2[4] += next[2];

        a3[2] += next[0];
        a3[3] += next[1];
        a3[4] += next[2];

        a3[2] += x == lab_end - 3 ? next[0] : *(next + 3);
        a3[3] += x == lab_end - 3 ? next[1] : *(next + 4);
        a3[4] += x == lab_end - 3 ? next[2] : *(next + 5);

        a1 *= 0.25f;
        a2 *= 0.25f;
        a3 *= 0.25f;
        a4 *= 0.25f;

        auto a21 = a2 - a1;
        auto a23 = a2 - a3;
        auto a43 = a4 - a3;
        auto a41 = a4 - a1;

        *area_start = delta(a21, a23) + delta(a43, a41);
    }   
}

npy_array<float> compute_area(const npy_array<float>& lab_image)
{
    npy_array<float> area{{lab_image.shape()[0], lab_image.shape()[1]}};

    std::vector<std::future<void>> futures{};
    futures.reserve(lab_image.shape()[0]);

    const auto area_stride = lab_image.shape()[1];
    const auto stride = lab_image.shape()[1] * lab_image.shape()[2];
    auto lab_start = lab_image.begin();
    auto lab_end = lab_start + stride;
    auto previous = lab_start;
    auto next = lab_end;
    auto area_start = area.begin();

    for(auto y = 0UL; y != lab_image.shape()[0]; y++)
    {
        futures.push_back(std::async(
            policy,
            parallel_area,
            lab_start, lab_end, previous, next, area_start
        ));

        previous = lab_start;
        lab_start = lab_end;
        lab_end += stride;
        
        if(y == lab_image.shape()[0] - 1) next = lab_start;
        else next = lab_end;

        next += area_stride;
    }

    for(auto& future : futures)
    {
        future.get();
    }

    return std::move(area);
}

npy_array<float> compute_seeds(const npy_array<float>& lab_image, const int K, const npy_array<float>& cumulative_area)
{
    // (y, x, l, a, b)
    npy_array<float> seeds{{size_t(K), 5}};

    const float step = (cumulative_area[cumulative_area.size() - 1] - cumulative_area[0]) / float(K - 1);
    std::vector<float> selected_area(size_t(K), 0.0f);

    for(size_t i = 0; i < selected_area.size() - 1; i++)
    {
        selected_area[i] = cumulative_area[0] + (i * step);
    }
    
    selected_area[K - 1] = cumulative_area[cumulative_area.size() - 1];
    
    for(auto i = 0UL, j = 0UL; i < selected_area.size(); i++)
    {
        while(j < cumulative_area.size() && cumulative_area[j] < selected_area[i])
        {
            j++;
        }

        const auto coord = ind2sub(j, cumulative_area.shape()[1]);
        const auto seed_index = i * seeds.shape()[1];

        seeds[seed_index] = float(coord.first);
        seeds[seed_index + 1] = float(coord.second);
        seeds[seed_index + 2] = lab_image[{coord.first, coord.second, 0}];
        seeds[seed_index + 3] = lab_image[{coord.first, coord.second, 1}];
        seeds[seed_index + 4] = lab_image[{coord.first, coord.second, 2}];
    }

    return std::move(seeds);
}

int main(int argc, char* argv[])
{
    timer profiler{};

    const int region_size = 25;
    const int max_iterations = 1;

    profiler.start();
    npy_array<uint8_t> rgb_image{"./4k.npy"};
    profiler.stop_and_print();

    profiler.start();
    npy_array<float> lab_image = rgb_to_lab(rgb_image);
    profiler.stop_and_print();

    profiler.start();
    npy_array<float> area = compute_area(lab_image);
    profiler.stop_and_print();

    profiler.start();
    npy_array<float> cumulative_area{area.shape()};
    std::partial_sum(area.cbegin(), area.cend(), cumulative_area.begin(), std::plus<float>());
    profiler.stop_and_print();

    const int K = (rgb_image.shape()[0] * rgb_image.shape()[1]) / (region_size * region_size);
    const float xi = cumulative_area[cumulative_area.size() - 1] * 4.0f / float(K);

    profiler.start();
    volatile npy_array<float> seeds = compute_seeds(lab_image, K, cumulative_area);
    profiler.stop_and_print();

    return EXIT_SUCCESS;
}