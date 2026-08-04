#include <Bench.hpp>

#include <Mala/Platform/ThreadPool.hpp>

#include <cstddef>
#include <vector>

using namespace Mala;

int main()
{
    std::size_t const n = std::size_t{1} << 24; // 16M floats = 64 MB per array
    std::vector<float> x(n, 1.0f);
    std::vector<float> y(n, 2.0f);
    std::vector<float> z(n, 0.0f);

    // saxpy: z = 2x + y. Reads 2 arrays, writes 1 => 3 * n * sizeof(float) bytes moved.
    double const serial = Bench::timeSeconds(5, [&] {
        for (std::size_t i = 0; i < n; ++i) {
            z[i] = 2.0f * x[i] + y[i];
        }
    });
    Bench::reportBandwidth("saxpy serial", serial, 3.0 * static_cast<double>(n) * sizeof(float));

    double const parallel = Bench::timeSeconds(5, [&] {
        Platform::parallelFor(0, n, 65536, [&](std::size_t i0, std::size_t i1) {
            for (std::size_t i = i0; i < i1; ++i) {
                z[i] = 2.0f * x[i] + y[i];
            }
        });
    });
    Bench::reportBandwidth("saxpy parallel", parallel, 3.0 * static_cast<double>(n) * sizeof(float));

    std::println("speedup: {:.2f}x on {} threads",
             serial / parallel, Platform::ThreadPool::global().size());
    return 0;
}
