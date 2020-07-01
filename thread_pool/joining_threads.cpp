#include "joining_threads.h"

joining_threads::joining_threads(std::vector<std::thread>& threads) : _threads{threads} {}

joining_threads::~joining_threads(){
    for(auto i = 0UL; i != _threads.size(); ++i){
        if(_threads[i].joinable()) _threads[i].join();
    }
}
