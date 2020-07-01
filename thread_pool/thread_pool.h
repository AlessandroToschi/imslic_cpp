#ifndef DD1A1879_FD2C_42FE_B19E_333239B046A3
#define DD1A1879_FD2C_42FE_B19E_333239B046A3

#include <thread>
#include <atomic>
#include <algorithm>
#include <vector>
#include <iostream>
#include <future>
#include <type_traits>
#include <functional>

#include "function_wrapper.h"
#include "threadsafe_queue.h"
#include "joining_threads.h"

class thread_pool
{
public:
    thread_pool();
    explicit thread_pool(const size_t threads_count);

    thread_pool(const thread_pool& other) = delete;
    thread_pool(thread_pool&& other) = delete;

    thread_pool& operator=(const thread_pool& other) = delete;
    thread_pool& operator=(thread_pool&& other) = delete;

    ~thread_pool();

    template<typename F>
    std::future<typename std::result_of<F()>::type> submit(F&& function);
    template<typename F, typename... Args>
    auto submit(F&& function, Args&&... args);
private:
    std::atomic_bool _run;
    threadsafe_queue<function_wrapper> _queue;
    std::vector<std::thread> _threads;
    joining_threads _joiner;

    void thread_loop();
};

#include "thread_pool.tcc"

#endif /* DD1A1879_FD2C_42FE_B19E_333239B046A3 */
