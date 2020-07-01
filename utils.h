#include<utility>

namespace imslic
{

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

    template<typename T>
    struct triplet
    {
        T x;
        T y;
        T z;
    };
    

}