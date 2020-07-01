#ifndef C8174308_6CB7_4BAE_AAA8_D87A24BC467A
#define C8174308_6CB7_4BAE_AAA8_D87A24BC467A

#include <mutex>
#include <condition_variable>
#include <queue>

template<typename T>
class threadsafe_queue
{
public:
    threadsafe_queue();

    void push(T value);
    void wait_and_pop(T& value);
    bool try_pop(T& value);
    bool empty() const;
private:
    mutable std::mutex _mutex;
    uint8_t padding[64UL - sizeof(std::mutex)];
    std::queue<T> _queue;
    std::condition_variable _condition_variable;
};

#include "threadsafe_queue.tcc"

#endif /* C8174308_6CB7_4BAE_AAA8_D87A24BC467A */
