/*
   A fixed-size thread pool + parallerFor.

   Rolling our own because parallel <execution> is not yet available.
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>


namespace Mala::Platform
{

class ThreadPool
{
public:
    explicit ThreadPool(size_t nthreads = 0);
    ~ThreadPool();

    ThreadPool(ThreadPool const&)            = delete;
    ThreadPool& operator=(ThreadPool const&) = delete;

    size_t size() const noexcept
    {
        return workers_.size();
    }

    // Submit a task; returns immediately. The task runs on some  worker thread.
    void enqueue(std::function<void()> task);

    // The process-wide pool, sized to hardware_concurrency. Constructed on first use.
    static ThreadPool& global();

private:
    void workerLoop();

    std::vector<std::thread>            workers_;
    std::queue<std::function<void()>>   tasks_;
    std::mutex                          mutex_;
    std::condition_variable             cond_;
    bool                                stop_ = false;
};

// Run Body(i0, i1) over a partition of [begin, end) to the global pool, blocking until all
// chunks finish. Chunks are at least `grain` wide and capped at the thread count. `body`
// must be safe to call concurrently on disjoint sub-ranges.
template <typename Callable>
void parallelFor(size_t begin, size_t end, size_t grain, Callable&& body)
{
    if (end <= begin)
        return;

    size_t const n = end - begin;

    ThreadPool& pool = ThreadPool::global();
    size_t const nthreads = pool.size();
    if (nthreads <= 1 || n <= grain) { // not worth scheduling
        body(begin, end);
        return;
    }

    size_t chunks = (n + grain - 1) / grain;
    if (chunks > nthreads)
        chunks = nthreads;
    
    size_t chunk = (n + chunks - 1) / chunks;

    // Materialize the (non-empty) ranges up front so `remaining` is exact.
    std::vector<std::pair<size_t, size_t>> ranges;
    ranges.reserve(chunks);
    for (size_t i0 = begin; i0 < end; i0 += chunk) {
        ranges.emplace_back(i0, std::min(end, i0 + chunk));
    }

    std::atomic<size_t> remaining { ranges.size() };
    std::mutex doneMutex;
    std::condition_variable done;

    for (auto [i0, i1]: ranges) {
        pool.enqueue([&, i0, i1] {
            body(i0, i1);
            if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard lock(doneMutex);
                done.notify_one();
            }
        });
    }

    std::unique_lock lock(doneMutex);
    done.wait(lock, [&] {
        return remaining.load(std::memory_order_acquire) == 0;
    });
}

} // namespace Mala::Platform
