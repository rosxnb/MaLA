#include <Mala/Platform/ThreadPool.hpp>


namespace Mala::Platform
{

ThreadPool::ThreadPool(size_t nthreads)
{
    if (nthreads == 0) {
        nthreads = std::thread::hardware_concurrency();
        if (nthreads == 0)
            nthreads = 1;
    }

    workers_.reserve(nthreads);
    for (size_t i = 0; i < nthreads; ++i) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }

    cond_.notify_all();
    for (auto& worker: workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::enqueue(std::function<void()> task)
{
    {
        std::lock_guard lock(mutex_);
        tasks_.push(std::move(task));
    }

    cond_.notify_one();
}

void ThreadPool::workerLoop()
{
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock lock(mutex_);
            cond_.wait(lock, [this] {
                return stop_ || !tasks_.empty();
            });

            if (stop_ && tasks_.empty())
                return;

            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

ThreadPool& ThreadPool::global()
{
    static ThreadPool pool;
    return pool;
}

} // namespace Mala::Platform
