#include <iostream>

//#include "npy_array/npy_array.h"

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

int main(int argc, char* argv[])
{
    return EXIT_SUCCESS;
}