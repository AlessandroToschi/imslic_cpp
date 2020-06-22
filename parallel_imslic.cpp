#include <iostream>
#include <future>
#include <cassert>
#include <valarray>
#include <chrono>

#include "npy_array/npy_array.h"

std::launch policy = std::launch::deferred;

class timer
{
    typedef std::chrono::high_resolution_clock::time_point time_point;

public:
    timer() : _start{}, _end{} {}
    void start()
    {
        _start = std::chrono::high_resolution_clock::now();
    }

    void stop_and_print(const std::string& message="")
    {
        _end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start);
        
        std::cout << message << " Duration: " << duration.count() << " ms" << std::endl;
    }

private:
    time_point _start;
    time_point _end;
};

struct region_distances
{
    std::vector<std::pair<size_t, size_t>> coords;
    std::unique_ptr<npy_array<float>> distances;
    size_t k;
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
        futures.push_back(std::async(
            policy, 
            parallel_rgb_to_lab, 
            rgb_start, rgb_end, lab_start
        ));

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

void parallel_area(const float* lab_start, const float* lab_end, const float* previous, const float* next, float* area_start, const size_t y)
{
    std::valarray<float> a1(0.0f, 5), a2(0.0f, 5), a3(0.0f, 5), a4(0.0f, 5);

    size_t xx = 0;

    for(auto x = lab_start, p = previous, n = next; x != lab_end; x +=3, p += 3, n += 3, area_start += 1, xx++)
    {
        a1[0] = 4.0f * float(y) - 2.0f;
        a1[1] = 4.0f * float(xx) - 2.0f;
        a1[2] = x[0];
        a1[3] = x[1];
        a1[4] = x[2];

        a2 = a1;
        a3 = a1;
        a4 = a1;

        a2[0] = 4.0f * float(y) + 2.0f;
        a2[1] = 4.0f * float(xx) - 2.0f;

        a3[0] = 4.0f * float(y) + 2.0f;
        a3[1] = 4.0f * float(xx) + 2.0f;

        a4[0] = 4.0f * float(y) - 2.0f;
        a4[1] = 4.0f * float(xx) + 2.0f;

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

        float d = delta(a21, a23) + delta(a43, a41);
        
        if(std::isnan(d))
        {
            std::cout << "ciao" << std::endl;
        }

        *area_start = d;
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
            lab_start, lab_end, previous, next, area_start, y
        ));

        previous = lab_start;
        lab_start = lab_end;
        lab_end += stride;

        area_start += area_stride;
    }

    for(auto& future : futures)
    {
        future.get();
    }

    return std::move(area);
}

npy_array<float> compute_seeds(const npy_array<float>& lab_image, const size_t K, const npy_array<float>& cumulative_area)
{
    // (y, x, l, a, b)
    npy_array<float> seeds{{K, 5}};

    const float step = (cumulative_area[cumulative_area.size() - 1] - cumulative_area[0]) / float(K - 1);
    std::vector<float> selected_area(K, 0.0f);

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

float compute_lambda(const float* seed_index, const npy_array<float>& area, const float xi, const size_t region_size)
{
    const size_t seed_y = size_t(seed_index[0]);
    const size_t seed_x = size_t(seed_index[1]);

    const size_t x_min = size_t(std::max(0L, long(seed_x) - long(region_size)));
    const size_t x_max = std::min(area.shape()[1] - 1UL, seed_x + region_size);
    const size_t y_min = size_t(std::max(0L, long(seed_y) - long(region_size)));
    const size_t y_max = std::min(area.shape()[0] - 1UL, seed_y + region_size);

    float sub_area = 0.0;

    for(size_t y = y_min; y <= y_max; y++)
    {
        const size_t start_index = sub2ind(x_min, y, area.shape()[1]);
        const size_t end_index = start_index + (x_max - x_min);

        sub_area += std::accumulate(&area[start_index], &area[end_index], 0.0f);
    }

    return std::sqrt(xi / sub_area);
}

npy_array<float> shortest_path(const int x_min, const int x_max, const int y_min, const int y_max, const int seed_x, const int seed_y, const npy_array<float>& lab_image)
{
    const float float_max = std::numeric_limits<float>::max();
    const int region_width = x_max - x_min + 1;
    const int region_height = y_max - y_min + 1;
    const int elements = region_width * region_height;

    npy_array<float> D{{size_t(elements)}};
    npy_array<int> V{{size_t(elements)}};

    std::fill(D.begin(), D.end(), float_max);
    std::fill(V.begin(), V.end(), 0);
    
    D[sub2ind(seed_x - x_min, seed_y - y_min, region_width)] = 0.0f;

    for(auto i = 0; i != elements - 1; i++)
    {
        int min_index = -1;
        float min_distance = float_max;

        for(auto j = 0; j != elements; j++)
        {
            if(V[j] == 0 && D[j] < min_distance)
            {
                min_distance = D[j];
                min_index = j;
            }
        }

        V[min_index] = 1;

        const auto xy_relative = ind2sub(min_index, region_width);
        const float* current_pixel = &lab_image[{size_t(y_min + xy_relative.first), size_t(x_min + xy_relative.first)}];

        for(auto dx = -1; dx <= 1; dx++)
        {
            for(auto dy = -1; dy <= 1; dy++)
            {
                if(0 <= xy_relative.second + dx && xy_relative.second + dx < region_width && 0 <= xy_relative.first + dy && xy_relative.first + dy < region_height)
                {
                    const float* next_pixel = &lab_image[{size_t(y_min + xy_relative.first + dy), size_t(x_min + xy_relative.first + dx)}];
                    const int next_index = sub2ind(xy_relative.second + dx, xy_relative.first + dy, region_width);

                    const float distance = std::sqrt(
                        std::pow(current_pixel[0] - next_pixel[0], 2.0f) + 
                        std::pow(current_pixel[1] - next_pixel[1], 2.0f) + 
                        std::pow(current_pixel[2] - next_pixel[2], 2.0f)
                    );

                    if(V[next_index] == 0 && D[min_index] != float_max && D[min_index] + distance < D[next_index])
                    {
                        D[next_index] = D[min_index] + distance;
                    }
                }
            }
        }
    }
    return std::move(D);
}

region_distances parallel_region_distance(const size_t k, const npy_array<float>& seeds, const npy_array<float>& lab_image, const npy_array<float>& area, const float xi, const size_t region_size)
{
    region_distances rd;

    const float* seed_index = &seeds[{k, 0}];

    const size_t y = size_t(seed_index[0]);
    const size_t x = size_t(seed_index[1]);

    const float lambda = compute_lambda(seed_index, area, xi, region_size);
    const size_t offset = lambda * region_size;

    const size_t x_min = size_t(std::max(0L, long(x) - long(offset)));
    const size_t x_max = std::min(area.shape()[1] - 1UL, x + offset);
    const size_t y_min = size_t(std::max(0L, long(y) - long(offset)));
    const size_t y_max = std::min(area.shape()[0] - 1UL, y + offset);

    const size_t region_width = x_max - x_min + 1;

    rd.k = k;
    rd.distances.reset(new npy_array<float>(std::move(shortest_path(x_min, x_max, y_min, y_max, x, y, lab_image))));
    rd.coords.reserve(rd.distances->size());

    for(auto i = 0UL; i != rd.distances->size(); i++)
    {
        auto xy_relative = ind2sub(i, region_width);
        rd.coords.push_back(std::make_pair(xy_relative.first + y_min, xy_relative.second + x_min));
    }

    return std::move(rd);
}

std::vector<std::pair<size_t, size_t>> find_orphanes(const npy_array<int>& labels)
{
    std::vector<std::pair<size_t, size_t>> orphanes{};

    for(auto i = 0UL; i != labels.size(); i++)
    {
        if(labels[i] == -1)
        {
            orphanes.push_back(ind2sub(i, labels.shape()[1]));
        }
    }

    return std::move(orphanes);
}  

int main(int argc, char* argv[])
{
    timer profiler{};

    const size_t region_size = 25;
    const size_t max_iterations = 1;

    profiler.start();
    npy_array<uint8_t> rgb_image{"./image.npy"};
    profiler.stop_and_print("RGB Loading");

    profiler.start();
    npy_array<float> lab_image = rgb_to_lab(rgb_image);
    profiler.stop_and_print("RGB 2 LAB");

    profiler.start();
    npy_array<float> area = compute_area(lab_image);
    profiler.stop_and_print("Area");

    profiler.start();
    npy_array<float> cumulative_area{area.shape()};
    std::partial_sum(area.cbegin(), area.cend(), cumulative_area.begin(), std::plus<float>());
    profiler.stop_and_print("Cumulative Area");

    const size_t K = (rgb_image.shape()[0] * rgb_image.shape()[1]) / (region_size * region_size);
    const float xi = cumulative_area[cumulative_area.size() - 1] * 4.0f / float(K);

    profiler.start();
    npy_array<float> seeds = compute_seeds(lab_image, K, cumulative_area);
    profiler.stop_and_print("Seeds");

    npy_array<int> labels{{rgb_image.shape()[0], rgb_image.shape()[1]}};
    npy_array<float> global_distances{labels.shape()};

    for(auto iteration = 0UL; iteration != max_iterations; iteration++)
    {
        profiler.start();

        timer inner_timer{};

        inner_timer.start();
        std::fill(global_distances.begin(), global_distances.end(), std::numeric_limits<float>::max());
        std::fill(labels.begin(), labels.end(), -1);
        inner_timer.stop_and_print("Distances and Labels Reset");

        inner_timer.start();

        std::vector<std::future<region_distances>> futures;
        futures.reserve(K);

        for(auto k = 0UL; k != K; k++)
        {
            futures.push_back(std::async(
                policy,
                parallel_region_distance,
                k, std::ref(seeds), std::ref(lab_image), std::ref(area), xi, region_size
            ));
        }

        for(auto& future : futures)
        {
            region_distances rd = std::move(future.get());

            for(auto i = 0UL; i != rd.coords.size(); i++)
            {
                auto& coord = rd.coords[i];
                float& distance = rd.distances->operator[](i);

                if(distance < global_distances[{coord.first, coord.second}])
                {
                    global_distances[{coord.first, coord.second}] = distance;
                    labels[{coord.first, coord.second}] = rd.k;
                }
            }
        }

        inner_timer.stop_and_print("Region Distances");

        inner_timer.start();

        auto orphanes = std::move(find_orphanes(labels));

        inner_timer.stop_and_print("Orphanes");

        profiler.stop_and_print("Iteration " + std::to_string(iteration));
    }

    return EXIT_SUCCESS;
}