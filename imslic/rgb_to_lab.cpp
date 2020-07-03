#include "rgb_to_lab.h"

inline void rgb_to_lab_base::scale_pixel(const uint8_t *pixel, float *scaled_pixel, const float scale) const{
    scaled_pixel[0] = float(pixel[0]) * scale;
    scaled_pixel[1] = float(pixel[1]) * scale;
    scaled_pixel[2] = float(pixel[2]) * scale;
}

inline void rgb_to_lab_base::inverse_gamma(const float *srgb_pixel, float *rgb_pixel) const{
    rgb_pixel[0] = srgb_pixel[0] <= 0.04045f ? srgb_pixel[0] * 0.077399381f : powf((srgb_pixel[0] + 0.055f) * 0.947867299f, 2.4f); 
    rgb_pixel[1] = srgb_pixel[1] <= 0.04045f ? srgb_pixel[1] * 0.077399381f : powf((srgb_pixel[1] + 0.055f) * 0.947867299f, 2.4f); 
    rgb_pixel[2] = srgb_pixel[2] <= 0.04045f ? srgb_pixel[2] * 0.077399381f : powf((srgb_pixel[2] + 0.055f) * 0.947867299f, 2.4f);
}

inline void rgb_to_lab_base::rgb_to_xyz(const float *rgb_pixel, float *xyz_pixel) const{
    xyz_pixel[0] = (0.412453f * rgb_pixel[0] + 0.35758f * rgb_pixel[1] + 0.180423f * rgb_pixel[2]) / 0.950456f;
    xyz_pixel[1] = 0.212671f * rgb_pixel[0] + 0.71516f * rgb_pixel[1] + 0.072169f * rgb_pixel[2];
    xyz_pixel[2] = (0.019334f * rgb_pixel[0] + 0.119193f * rgb_pixel[1] + 0.950227f * rgb_pixel[2]) / 1.088754f;
}

inline void rgb_to_lab_base::lab_function(const float *xyz_pixel, float *lab_pixel) const{
    float fxfyfz_pixel[3];

    fxfyfz_pixel[0] = xyz_pixel[0] > 0.008856f ? std::cbrtf(xyz_pixel[0]) : 7.787f * xyz_pixel[0] + 0.137931034f;
    fxfyfz_pixel[1] = xyz_pixel[1] > 0.008856f ? std::cbrtf(xyz_pixel[1]) : 7.787f * xyz_pixel[1] + 0.137931034f;
    fxfyfz_pixel[2] = xyz_pixel[2] > 0.008856f ? std::cbrtf(xyz_pixel[2]) : 7.787f * xyz_pixel[2] + 0.137931034f;
    
    lab_pixel[0] = xyz_pixel[1] > 0.008856f ? 116.0f * fxfyfz_pixel[1] - 16.0f : 903.3f * fxfyfz_pixel[1];
    lab_pixel[1] = 500.0f * (fxfyfz_pixel[0] - fxfyfz_pixel[1]);
    lab_pixel[2] = 200.0f * (fxfyfz_pixel[1] - fxfyfz_pixel[2]);
}



npy_array<float> sequential_rgb_to_lab::operator()(const npy_array<uint8_t> &rgb_image) const{
    npy_array<float> lab_image{rgb_image.shape()};

    float scaled_rgb_pixel[3];
    float rgb_pixel[3];
    float xyz_pixel[3];

    for(auto y = 0UL; y != rgb_image.shape()[0]; ++y){
        for(auto x = 0UL; x != rgb_image.shape()[1]; ++x){
            rgb_to_lab_base::scale_pixel(&rgb_image[{y, x, 0UL}], scaled_rgb_pixel, 0.003921569f);
            rgb_to_lab_base::inverse_gamma(scaled_rgb_pixel, rgb_pixel);
            rgb_to_lab_base::rgb_to_xyz(rgb_pixel, xyz_pixel);
            rgb_to_lab_base::lab_function(xyz_pixel, &lab_image[{y, x, 0UL}]);
        }
    }

    return std::move(lab_image);
}

parallel_rgb_to_lab::parallel_rgb_to_lab(thread_pool &pool)
    : _pool{pool} {}

npy_array<float> parallel_rgb_to_lab::operator()(const npy_array<uint8_t> &rgb_image) const{
    npy_array<float> lab_image{rgb_image.shape()};
    
    std::vector<std::future<void>> futures;
    futures.reserve(rgb_image.shape()[0]);

    const auto stride = rgb_image.shape()[1] * rgb_image.shape()[2];
    auto rgb_start = rgb_image.begin();
    auto rgb_end = rgb_start + stride;
    auto lab_start = lab_image.begin();

    for(auto y = 0UL; y != rgb_image.shape()[0]; ++y)
    {
        futures.push_back(_pool.submit([=](){
            float* out = lab_start;
            float scaled_rgb_pixel[3];
            float rgb_pixel[3];
            float xyz_pixel[3];

            for(auto i = rgb_start; i != rgb_end; i += 3, out += 3){
                rgb_to_lab_base::scale_pixel(i, scaled_rgb_pixel, 0.003921569f);
                rgb_to_lab_base::inverse_gamma(scaled_rgb_pixel, rgb_pixel);
                rgb_to_lab_base::rgb_to_xyz(rgb_pixel, xyz_pixel);
                rgb_to_lab_base::lab_function(xyz_pixel, out);
            }
        }));
    }

    for(auto i = 0UL; i != futures.size(); ++i)
    {
        futures[i].get();
    }

    return std::move(lab_image);
}
