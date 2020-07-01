#ifndef A45D6E3F_B1C3_4289_9D8D_DA73C78D9B40
#define A45D6E3F_B1C3_4289_9D8D_DA73C78D9B40

#include <memory>

class function_wrapper
{
public:
    function_wrapper() = default;
    template<typename F>
    function_wrapper(F&& function);
    function_wrapper(function_wrapper&& other);
    function_wrapper(const function_wrapper& other) = delete;

    void operator()();
    function_wrapper& operator=(function_wrapper&& other);
    function_wrapper& operator=(const function_wrapper& other) = delete;
private:
    struct function_base
    {
        virtual void call() = 0;
        virtual ~function_base() {};
    };
    template<typename F>
    struct function_impl : function_base
    {
        F _function;
        function_impl(F&& function) : _function{std::move(function)} {}
        void call() {_function();}
    };
    std::unique_ptr<function_base> _function_pointer;
};

#include "function_wrapper.tcc"

#endif /* A45D6E3F_B1C3_4289_9D8D_DA73C78D9B40 */
