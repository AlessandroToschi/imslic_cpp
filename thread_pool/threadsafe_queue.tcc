#ifndef F41A7DA8_0DC4_49DB_BFFA_070CEFFFD4C4
#define F41A7DA8_0DC4_49DB_BFFA_070CEFFFD4C4

#include "threadsafe_queue.h"

template<typename T>
threadsafe_queue<T>::threadsafe_queue(){}

template<class T> 
void threadsafe_queue<T>::push(T value){
    std::lock_guard<std::mutex> lock{_mutex};
    _queue.push(std::move(value));
    _condition_variable.notify_one();
}

template<class T> 
void threadsafe_queue<T>::wait_and_pop(T &value){
    std::unique_lock<std::mutex> lock{_mutex};
    _condition_variable.wait(lock, [this]{return !this->_queue.empty();});
    value = std::move(_queue.front());
    _queue.pop();
}

template<class T> 
bool threadsafe_queue<T>::try_pop(T &value){
    std::lock_guard<std::mutex> lock{_mutex};
    if(_queue.empty()) return false;
    value = std::move(_queue.front());
    _queue.pop();
    return true;
}


template<class T> 
bool threadsafe_queue<T>::empty() const{
    std::lock_guard<std::mutex> lock{_mutex};
    return _queue.empty();
}





#endif /* F41A7DA8_0DC4_49DB_BFFA_070CEFFFD4C4 */
