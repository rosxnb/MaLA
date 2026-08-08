#include <MalaTest.hpp>

#include <Mala/Core/Shape.hpp>
#include <Mala/Core/Tensor.hpp>

using namespace Mala;

// –– Shape helpers ––––––––––––––––––––––––––––––––––––––

MALA_TEST(shapeHelpers)
{
    Dims const shape{2, 3, 4};
    CHECK(numElements(shape) == 24);

    Dims const strides = contiguousStrides(shape);
    CHECK(strides[0] == 12);
    CHECK(strides[1] == 4);
    CHECK(strides[2] == 1);
    CHECK(isContiguous(shape, strides));

    // rank-0 scalar has 1 element
    CHECK(numElements(Dims{}) == 1);
}

MALA_TEST(broadcastRules)
{
    Dims const out = broadcastShapes(Dims{3, 1, 5}, Dims{4, 5});
    CHECK(out.rank() == 3);
    CHECK(out[0] == 3);
    CHECK(out[1] == 4);
    CHECK(out[2] == 5);

    CHECK(broadcastShapes(Dims{1}, Dims{7})[0] == 7);
    CHECK_THROWS(broadcastShapes(Dims{3}, Dims{4})); // 3 vs 4, neither is 1
}

// –– Factories ––––––––––––––––––––––––––––––––––––––––––––

MALA_TEST(factoriesAndAccess)
{
    auto zeros = Tensor<float>::zeros(Dims{2, 2});
    CHECK_CLOSE(zeros(0, 0), 0.0, 0.0);
    CHECK_CLOSE(zeros(1, 1), 0.0, 0.0);

    auto full = Tensor<float>::full(Dims{2, 3}, 7.0f);
    CHECK_CLOSE(full(1, 2), 7.0, 0.0);
    CHECK(full.numel() == 6);
    CHECK(full.isContiguousLayout());

    auto ramp = Tensor<int>::arange(0, 5);
    CHECK(ramp.numel() == 5);
    CHECK(ramp(0) == 0);
    CHECK(ramp(4) == 4);
}

// –– Views –––––––––––––––––––––––––––––––––––––––––––––––––––

MALA_TEST(viewSharesStorage)
{
    auto flat = Tensor<float>::arange(0.0f, 6.0f); // shape {6}: 0,1,2,3,4,5
    auto grid = flat.view(Dims{2, 3});             // rows [0,1,2] , [3,4,5]

    CHECK(grid.rank() == 2);
    CHECK_CLOSE(grid(1, 2), 5.0, 0.0);
    CHECK(grid.storage().get() == flat.storage().get()); // same underlying buffer

    grid(0, 0) = 42.0f;                 // mutate through the view ...
    CHECK_CLOSE(flat(0), 42.0, 0.0);    // ... visible in the original (aliasing)

    CHECK_THROWS(flat.view(Dims{4, 4})); // wrong element count
}

MALA_TEST(transposeIsStridedView)
{
    auto grid = Tensor<float>::arange(0.0f, 6.0f).view(Dims{2, 3}); // strides {3,1}
    auto t = grid.transpose(0, 1);                                  // shape {3,2}

    CHECK(t.shape()[0] == 3);
    CHECK(t.shape()[1] == 2);
    CHECK(t.strides()[0] == 1); // strides swapped
    CHECK(t.strides()[1] == 3);
    CHECK(!t.isContiguousLayout());

    CHECK_CLOSE(t(0, 1), grid(1, 0), 0.0); // both index the same element (== 3)
    CHECK_CLOSE(t(2, 1), grid(1, 2), 0.0); // == 5
}

MALA_TEST(permuteReordersAxes)
{
    auto a = Tensor<int>::arange(0, 24).view(Dims{2, 3, 4});
    auto p = a.permute({2, 0, 1}); // new shape {4,2,3}

    CHECK(p.shape()[0] == 4);
    CHECK(p.shape()[1] == 2);
    CHECK(p.shape()[2] == 3);
    CHECK(p(3, 1, 2) == a(1, 2, 3)); // permuted index maps back
}

MALA_TEST(sliceAndNarrow)
{
    auto a = Tensor<int>::arange(0, 10);
    auto s = a.slice(0, 2, 8, 2); // indices 2,4,6
    CHECK(s.shape()[0] == 3);
    CHECK(s(0) == 2);
    CHECK(s(1) == 4);
    CHECK(s(2) == 6);

    auto n = a.narrow(0, 3, 4); // indices 3,4,5,6
    CHECK(n.shape()[0] == 4);
    CHECK(n(0) == 3);
    CHECK(n(3) == 6);

    CHECK_THROWS(a.slice(0, 0, 20)); // stop out of range
}

MALA_TEST(expandUsesZeroStride)
{
    auto col = Tensor<float>::arange(0.0f, 3.0f).view(Dims{3, 1}); // [[0],[1],[2]]
    auto e = col.broadcastTo(Dims{3, 4});

    CHECK(e.shape()[0] == 3);
    CHECK(e.shape()[1] == 4);
    CHECK(e.strides()[1] == 0); // broadcast axis has stride 0

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            CHECK_CLOSE(e(i, j), static_cast<double>(i), 0.0);
        }
    }

    CHECK_THROWS(col.broadcastTo(Dims{2, 4})); // 3 cannot broadcast to 2
}

MALA_TEST(squeezeAndUnsqueeze)
{
    auto a = Tensor<float>::ones(Dims{1, 3, 1});
    auto s = a.squeeze();
    CHECK(s.rank() == 1);
    CHECK(s.shape()[0] == 3);

    auto u = s.unsqueeze(0);
    CHECK(u.rank() == 2);
    CHECK(u.shape()[0] == 1);
    CHECK(u.shape()[1] == 3);

    auto one = a.squeeze(0); // drop only axis 0 -> {3,1}
    CHECK(one.rank() == 2);
    CHECK(one.shape()[0] == 3);
    CHECK(one.shape()[1] == 1);
    CHECK_THROWS(a.squeeze(1)); // axis 1 has extent 3
}

MALA_TEST(flipReversesAxis)
{
    auto a = Tensor<int>::arange(0, 4);
    auto f = a.flip(0);
    CHECK(f(0) == 3);
    CHECK(f(1) == 2);
    CHECK(f(2) == 1);
    CHECK(f(3) == 0);
    CHECK(a(0) == 0); // original untouched
}

// –– Copies ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

MALA_TEST(cloneIsIndependentAndContiguous)
{
    auto grid = Tensor<float>::arange(0.0f, 6.0f).view(Dims{2, 3});
    auto t = grid.transpose(0, 1); // non-contiguous {3,2}
    auto c = t.clone();

    CHECK(c.isContiguousLayout());
    CHECK(c.shape()[0] == 3);
    CHECK(c.shape()[1] == 2);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            CHECK_CLOSE(c(i, j), t(i, j), 0.0); // values preserved through the gather
        }
    }

    c(0, 0) = -1.0f;                    // clone owns separate storage ...
    CHECK_CLOSE(grid(0, 0), 0.0, 0.0);  // ... original is unaffected
}

MALA_TEST(contiguousMethodMatchesClone)
{
    auto t = Tensor<float>::arange(0.0f, 6.0f).view(Dims{2, 3}).transpose(0, 1);
    auto c = t.contiguous();
    CHECK(c.isContiguousLayout());
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            CHECK_CLOSE(c(i, j), t(i, j), 0.0);
        }
    }
}

// –– Mdspan bridge ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

MALA_TEST(asMdspanMatchesAccess)
{
    auto grid = Tensor<float>::arange(0.0f, 6.0f).view(Dims{2, 3});
    auto ms = grid.asMdspan<2>();
    CHECK(ms.extent(0) == 2);
    CHECK(ms.extent(1) == 3);
    CHECK_CLOSE((ms[1, 2]), grid(1, 2), 0.0); // C++23 multidimensional subscript

    auto t = grid.transpose(0, 1); // positive strides {1,3} -> valid for mdspan
    auto mst = t.asMdspan<2>();
    CHECK_CLOSE((mst[0, 1]), t(0, 1), 0.0);

    CHECK_THROWS(grid.asMdspan<3>()); // rank mismatch
}
