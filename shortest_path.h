#ifndef CD452B50_1DE1_4640_8EBE_C9559B010C66
#define CD452B50_1DE1_4640_8EBE_C9559B010C66

#include <queue>

#include "utils.h"
#include "npy_array/npy_array.h"

namespace imslic
{
    class shortest_path
    {
    public:
        explicit shortest_path(const npy_array<float>* image) : 
            _image{image}, 
            _neighbor_distances{{image->shape()[0] * image->shape()[1], 9}},
            _initialized{false} {}
        
        ~shortest_path() = default;

        npy_array<float> operator()(
            const size_t seed_x, 
            const size_t seed_y, 
            const size_t x_min,
            const size_t x_max,
            const size_t y_min,
            const size_t y_max
        )
        {
            if(!_initialized)
            {
                this->initialize_neighbor_distances();

                _initialized = true;
            }

            const float float_max = std::numeric_limits<float>::max();
            const auto region_width = x_max - x_min + 1;
            const auto region_height = y_max - y_min + 1;
            const auto elements = region_width * region_height;

            npy_array<float> distances{{elements}};
            std::fill(distances.begin(), distances.end(), float_max);

            distances[sub2ind(seed_x - x_min, seed_y - y_min, region_width)] = 0.0f;

            std::priority_queue<pixel_queue, std::vector<pixel_queue>, std::greater<pixel_queue>> queue{};
            queue.push({seed_x, seed_y, 0.0f});
        
            while(!queue.empty())
            {
                const auto current_pixel = queue.top();
                queue.pop();

                const auto local_x = current_pixel.x - x_min;
                const auto local_y = current_pixel.y - y_min;
                const auto local_index = sub2ind(local_x, local_y, region_width);
                const auto global_index = sub2ind(current_pixel.x, current_pixel.y, _image->shape()[1]);

                for(auto i = 0UL; i != _neighbor_distances.shape()[1]; i++)
                {
                    if(_neighbor_distances[{global_index, i}] != 0.0f)
                    {
                        auto relative_offset = ind2sub(long(i), 3L);
                        relative_offset.first -= 1L;
                        relative_offset.second -= 1L;

                        const auto neighbor_x = size_t(relative_offset.second + long(current_pixel.x));
                        const auto neighbor_y = size_t(relative_offset.first + long(current_pixel.y));

                        if(check_boundaries(x_min, x_max + 1UL, neighbor_x) && check_boundaries(y_min, y_max + 1UL, neighbor_y))
                        {
                            const auto neighbor_local_index = sub2ind(neighbor_x - x_min, neighbor_y - y_min, region_width);

                            if(distances[neighbor_local_index] > distances[local_index] + _neighbor_distances[{global_index, i}])
                            {
                                distances[neighbor_local_index] = distances[local_index] + _neighbor_distances[{global_index, i}];
                                queue.push({neighbor_x, neighbor_y, distances[neighbor_local_index]});
                            }
                        }
                    }
                }
            }

            return std::move(distances);
        }   
    private:
        void initialize_neighbor_distances()
        {
            std::fill(_neighbor_distances.begin(), _neighbor_distances.end(), 0.0f);

            for(auto y = 0UL; y != _image->shape()[0]; y++)
            {
                for(auto x = 0UL; x != _image->shape()[1]; x++)
                {
                    const auto linear_index = sub2ind(y, x, _image->shape()[1]);
                    const triplet<float>* pixel = reinterpret_cast<const triplet<float>*>(
                        &_image->operator[]({y, x, 0UL})
                    );

                    if(x < _image->shape()[1] - 1UL)
                    {
                        float d = this->pixels_distance(
                            1.0f, 
                            0.0f, 
                            pixel, 
                            reinterpret_cast<const triplet<float>*>(&_image->operator[]({y, x + 1UL, 0UL}))
                        );
                        _neighbor_distances[{linear_index, 5UL}] = d;
                        _neighbor_distances[{sub2ind(y, x + 1UL, _image->shape()[1]), 3UL}] = d;

                        if(y < _image->shape()[0] - 1UL)
                        {
                            d = this->pixels_distance(
                                1.0f, 
                                1.0f, 
                                pixel, 
                                reinterpret_cast<const triplet<float>*>(&_image->operator[]({y + 1UL, x + 1UL, 0UL}))
                            );
                            _neighbor_distances[{linear_index, 8UL}] = d;
                            _neighbor_distances[{sub2ind(y + 1UL, x + 1UL, _image->shape()[1]), 0UL}] = d;
                        }
                    }

                    if(y < _image->shape()[0] - 1UL)
                    {
                        float d = this->pixels_distance(
                            0.0f, 
                            1.0f, 
                            pixel, 
                            reinterpret_cast<const triplet<float>*>(&_image->operator[]({y + 1UL, x, 0UL}))
                        );
                        _neighbor_distances[{linear_index, 7UL}] = d;
                        _neighbor_distances[{sub2ind(y + 1UL, x, _image->shape()[1]), 1UL}] = d;

                        if(x > 0)
                        {
                            d = this->pixels_distance(
                                -1.0f, 
                                1.0f, 
                                pixel, 
                                reinterpret_cast<const triplet<float>*>(&_image->operator[]({y + 1UL, x - 1UL, 0UL}))
                            );

                            _neighbor_distances[{linear_index, 6UL}] = d;
                            _neighbor_distances[{sub2ind(y + 1UL, x - 1UL, _image->shape()[1]), 2UL}] = d;
                        }
                    }
                }
            }
        }

        inline float pixels_distance(
            const float dx, 
            const float dy, 
            const triplet<float>* a, 
            const triplet<float>* b
        )
        {
            return std::sqrt(
                std::pow(dx, 2.0f) +
                std::pow(dy, 2.0f) + 
                std::pow(a->x - b->x, 2.0f) +
                std::pow(a->y - b->y, 2.0f) +
                std::pow(a->z - b->z, 2.0f)
            );
        }

        struct pixel_queue
        {
            size_t x;
            size_t y;
            float distance;

            bool operator>(const pixel_queue& other) const
            {
                return distance > other.distance;
            }
        };

        

        const npy_array<float>* _image;
        npy_array<float> _neighbor_distances;
        bool _initialized;
    };

    
}

#endif /* CD452B50_1DE1_4640_8EBE_C9559B010C66 */
