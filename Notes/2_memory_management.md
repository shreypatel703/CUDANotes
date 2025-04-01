# CUDA Memory Management

## CUDA Memory Heirarchy

### Overview


| Memory Type | Scope       | Latency      | Size    | Best Usage                          |
| ------------- | ------------- | -------------- | --------- | ------------------------------------- |
| Global      | All Threads | High         | Large   | general data storage                |
| Shared      | Block wide  | Low          | Limited | fast acess data shared by block     |
| Local       | Per Thread  | High         | Small   | spillover from registers (avoid)    |
| Constant    | All Threads | Low (cached) | Small   | Read heavy operations, limit writes |

#### 1. Global Memory

* Accessible by all threads across all thread blocks.
* High latency (~400-600 clock cycles).
* Large capacity (typically several GBs in modern GPUs).
* No automatic caching (though modern GPUs have L2 caches).

#### 2. Shared Memory

* **Characteristics:**

  * Low latency (~100x faster than global memory).
  * Shared within a thread block.
  * Limited in size (typically 48KB per multiprocessor).
  * Explicitly managed by the programmer.
* **Best Practices:**

  * Use shared memory as a **software-managed cache** for global memory.
  * Reduce bank conflicts (accesses should be evenly distributed across memory banks).

#### 3. Local Memory

* **Characteristics:**
  * **Private** to each thread.
  * Stored in **global memory** (despite the name “local”).
  * Used for spilling registers or dynamically indexed arrays.
* **Best Practices:**
  * Avoid excessive use (it slows down performance due to high latency).
  * Use registers whenever possible.

```cpp
__global__ void local_memory_example() {
    int localVar = threadIdx.x;  // This is stored in a register (fast)
    __shared__ int sharedVar;    // Shared memory (fast)
    int array[10];               // May spill to local memory (slow)
}
```

#### 4.Constant Memory

* **Characteristics:**
  * **Read-only** and cached.
  * Optimized for broadcasting the same data across all threads.
  * **Limited size (~64KB total)** .
* **Best Practices:**
  * Use for frequently accessed read-only data (e.g., coefficients, lookup tables).
  * Avoid excessive access divergence (all threads should access the same location simultaneously).
* Example:

```cpp
__constant__ int constData[256];  // Declared in global scope

__global__ void constant_memory_example(int *arr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    arr[idx] *= constData[threadIdx.x];  // Reading from constant memory
}

```

The entire warp can read the same constant value in one cycle, making it highly efficient.

## Memory Allocation and Transfers

`cudaMalloc(void *ptr, size_t)` - Allocates memory on device

`cudaMemcpy(void* dst, void* src, size_t size, cudaMemcpyKind kind)` = copies data between host and device

- kind = `cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`

`cudaFree(void* ptr)` = Free Memory

`cudaMallocHost(void *ptr, size_t)` - Allocates pinned memory

* **Normal host memory** resides in pageable RAM, which can be swapped out, leading to slow data transfers.
* **Pinned memory** is page-locked, allowing direct memory access (DMA) for faster async CPU-GPU transfers
