#include "thread_pool.h"

thread_pool::thread_pool() : thread_pool(std::max(2U, std::thread::hardware_concurrency())){}

thread_pool::thread_pool(const size_t threads_count) : _run{true}, _joiner{_threads}{
    try{
        for(auto i = 0UL; i != threads_count; ++i){
            _threads.emplace_back(&thread_pool::thread_loop, this);
        }
    }
    catch(const std::exception& exception){
        _run.store(false);
        throw exception;
    }
}

void thread_pool::thread_loop(){
    while(_run.load()){
        function_wrapper task;
        if(_queue.try_pop(task)){
            task();
        }
        else{
            std::this_thread::yield();
        }
        
    }
}

thread_pool::~thread_pool(){
    _run.store(false);
}