# MaLA - Master Linear Algebra

A from scratch, zero-third-party **linear algebra + automatic-differentiation engine** built
to learn internals of deep-learning frameworks and to be the quality of production grade.
Inspirations taken from BLAS/NumPy/autograd.

> **Goal:** to build the substrate for training neural networks.

---

## Architecture at a glance

```
L5  autograd / nn   Tensor(reuires_grad), Module, Optimizer            - builds a DAG, backprops
L4  ops             add, matmul, softmax, conv2d, ...                  - forward + attach gradFn
L3  kernels         gemm, ewise, reduce, im2col, ...                   - the hot loops (SIMD + threads)
L2  tensor / view   shape, strides, dtype, broadcasting                - metadata over storage
L1  storage         aligned, ref-counted, tail-padded bytes            - owns memory
L0  platform        aligned alloc, CPU detect, simd<T, W>, thread pool
```

**Notes:**

- Each layer depends only on the ones below it.
- L1-L3 have no idea of autograd existence. MaLA can be used as BLAS/NumPy style library without graph overhead.
  Autograd is additive.
- L3 kernels operate on plain pointers + shape/stride, never on the rich Tensor type.
  This keeps them trivially testable, SIMD-friendly, and reusable across dtypes.
- L4 ops are thin: compute forward via L3, then (if any input needs grad) attach a `gradFn` node capturing
  exactly what backward needs.
