#include "function_wrapper.h"

function_wrapper::function_wrapper(function_wrapper &&other) :
    _function_pointer{std::move(other._function_pointer)} {}

void function_wrapper::operator()(){
    _function_pointer->call();
}

function_wrapper& function_wrapper::operator=(function_wrapper &&other){
    _function_pointer = std::move(other._function_pointer);
    return *this;
}

