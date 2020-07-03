#include "npy_array.h"
#include "thread_pool.h"
#include "rgb_to_lab.h"

int main(int, char**)
{
    npy_array<uint8_t> rgb_image{"4k.npy"};

    thread_pool pool{};

    parallel_rgb_to_lab p_rgb_to_lab{pool};
    
    auto lab_image = p_rgb_to_lab(rgb_image);


    return EXIT_SUCCESS;
}