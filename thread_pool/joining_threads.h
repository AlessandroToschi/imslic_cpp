#ifndef BBE1171F_E394_49D9_A6FD_8B3519721D15
#define BBE1171F_E394_49D9_A6FD_8B3519721D15

#include <vector>
#include <thread>

class joining_threads
{
public:
    explicit joining_threads(std::vector<std::thread>& threads);
    ~joining_threads();

private:
    std::vector<std::thread>& _threads;
};
#endif /* BBE1171F_E394_49D9_A6FD_8B3519721D15 */
