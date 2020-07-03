#include "npy_array.h"
#include "thread_pool.h"

class rgb_to_lab_base
{
public:
    virtual ~rgb_to_lab_base(){};
    virtual npy_array<float> operator()(const npy_array<uint8_t>& rgb_image) const = 0;

protected:
    void scale_pixel(const uint8_t* pixel, float* scaled_pixel, const float scale) const ;
    void inverse_gamma(const float* srgb_pixel, float* rgb_pixel) const;
    void rgb_to_xyz(const float* rgb_pixel, float* xyz_pixel) const;
    void lab_function(const float* xyz_pixel, float* lab_pixel) const;
};

class sequential_rgb_to_lab : rgb_to_lab_base
{
public:
    npy_array<float> operator()(const npy_array<uint8_t>& rgb_image) const;
};


class parallel_rgb_to_lab : rgb_to_lab_base
{
public:
    explicit parallel_rgb_to_lab(thread_pool& pool);

    npy_array<float> operator()(const npy_array<uint8_t>& rgb_image) const;

private:
    thread_pool& _pool;
};
