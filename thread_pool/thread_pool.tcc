#ifndef B9762DCF_91E1_4686_9CFB_7E1435C6882B
#define B9762DCF_91E1_4686_9CFB_7E1435C6882B

#include "thread_pool.h"

template<typename F> 
std::future<typename std::result_of<F()>::type> thread_pool::submit(F&& function){
    typedef typename std::result_of<F()>::type result_type;
    std::packaged_task<result_type()> task{std::move(function)};
    std::future<result_type> future{task.get_future()};
    _queue.push(std::move(task));
    return future;
}


template<typename F, typename... Args> 
auto thread_pool::submit(F&& function, Args&&... args){
    
    return this->submit(std::bind(std::forward<F>(function), std::forward<Args>(args)...));
}



#endif /* B9762DCF_91E1_4686_9CFB_7E1435C6882B */
