#ifndef E19D450B_FFC0_4C24_B56F_CD325E54B08B
#define E19D450B_FFC0_4C24_B56F_CD325E54B08B

#include "function_wrapper.h"

template<class F> 
function_wrapper::function_wrapper(F&& function) :
    _function_pointer{new function_impl<F>{std::move(function)}} {}


#endif /* E19D450B_FFC0_4C24_B56F_CD325E54B08B */
