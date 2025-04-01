# ``cppThread Synchronization

## Thread Divergence

* Thread Divergence happens when threads in a warp take different execution paths (caused by conditionals like if, switch, while)
* When threads in the same warp diverge this causes serialization and slows down the process

Divergent Example:

```cpp
__global__ void divergentKernel(int *arr, int *out) {
    int tid = threadIdx.x;
    if (tid % 2 == 0) {
        out[tid] = arr[tid] * 2; // Path A
    } else {
        out[tid] = arr[tid] + 1; // Path B
    }
}
```

You can minimize divergence by restructuring the code to be branch free:

```cpp
__global__ void optimizedKernel(int *arr, int *out) {
    int tid = threadIdx.x;
    int val = arr[tid];
    int isEven = tid % 2 == 0;
    out[tid] = isEven * (val * 2) + (!isEven) * (val + 1);
}
```

## Thread Synchronization

`__syncthreads();` - waits for all threads in the block to catch up before continuing

## Atomic Operations

When multiple threads need to update the same memory location, atomic operations ensure correctness **without explicit synchronization**.

* Arithmetic
  * atomicAdd(int *addr, int val);
  * atomicSub(int *addr, int valsum, va);
  * atomicExch(int *addr, int val)
  * atomicMin(int *addr, int val)
  * atomicMax(int *addr, int val)
* Logical and Bitwise Operations:
  * atomicAnd(int *addr, int val) //bitwise And
  * atomicOr(int *addr, int val)  //bitwise Or
  * atomicXor(int *addr, int val) //bitwise Xor
  * atomicCAS(int *addr, int val) //Compare and Swap


##### **Performance Trade-offs**

* **Advantages** :
  * Ensures correctness without race conditions.
  * No need for `__syncthreads()`, making it useful for global memory operations.
* **Disadvantages** :
  * **Serialization** : Only one thread can modify the value at a time, creating contention.
  * **Lower throughput** compared to parallel reduction methods.
