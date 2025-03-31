### **Topics to Learn**

#### **1. CUDA Architecture Overview**

- **SIMT (Single Instruction Multiple Threads) execution model**
- **GPU execution hierarchy** (Threads, Warps, Blocks, and Grids)
- How **warps** execute in **lock-step**

#### **2. Setting Up the CUDA Environment**

- Install **CUDA Toolkit & Nsight Compute**
- Configure **NVCC compiler**
- Write a **basic CUDA program**

#### **3. Launching a CUDA Kernel**

- CUDA kernel function syntax (`__global__` keyword)
- Understanding **threadIdx, blockIdx, blockDim, gridDim**

### **Project: Vector Addition in CUDA**

- Implement **parallel vector addition** (`C[i] = A[i] + B[i]`).
- Use **grid-stride loops** to handle large data sizes efficiently.

# Intro to CUDA

## What is CUDA

CUDA (Compute Unified Device Architecture) is a parallel computing platform and API developed by **NVIDIA** that enables developers to leverage the **massively parallel** computing power of **GPUs**

CUDA follows a **heterogeneous computing** model, where a program runs on both:

1. **CPU (Host)** – Handles sequential and control-flow logic.
2. **GPU (Device)** – Executes massively parallel workloads.

Kernels - functions that run in parallel across many threads on the GPU.

#### **Key Components of CUDA**

1. **Threads** : The smallest execution unit in CUDA. Thousands of threads run in parallel.
2. **Thread Blocks** : A group of threads that execute the same kernel function.
3. **Warps**: Group of threads inside of a block (32 threads for NVIDIA GPUs) that execute SIMULTANEOUSLY  on the GPU.
   - GPU schedules and executes warps not individual threads
   - All threads in a warp execute the same instruction at a time (SIMT execution)
   - Warp Divergence: If threads within a warp take different execution paths (e.g., an if-else branch), they serialize, reducing efficiency.
4. **Grid** : A collection of thread blocks. The grid organizes how threads are mapped to the GPU.
5. **Memory Hierarchy** :
   * **Global Memory** : Accessible by all threads, but slow.
   * **Shared Memory** : Shared within a thread block, much faster than global memory.
   * **Registers** : Fastest memory, used for per-thread storage.
   * **Constant Memory** : Read-only memory optimized for frequent reads.

##### Blocks vs Warps Example:

* If we launch a kernel with <<<4, 128>>>:
* Each block has 128 threads.
* Since a warp contains 32 threads, each block consists of 128 / 32 = 4 warps.
* Total warps: 4 (blocks) × 4 (warps per block) = 16 warps

## Using CUDA

### Basic Workflow

1. **Allocate memory on the GPU** (`cudaMalloc`).
2. **Copy data from CPU to GPU** (`cudaMemcpy`).
3. **Launch a kernel** (`func<<<block_per_grid, threads_per_block>>>()`).
4. **Copy results back to CPU** (`cudaMemcpy`).
5. **Free GPU memory** (`cudaFree`)

### Vector Addition Example

```
#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel function to add two arrays
__global__ void add(int *a, int *b, int *c, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int size = 100;
    int bytes = size * sizeof(int);

    // Allocate memory on host
    int *h_a = new int[size];
    int *h_b = new int[size];
    int *h_c = new int[size];

    // Initialize data
    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // 1. Allocate memory on device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 2. Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 3. Launch kernel with 10 blocks, 10 threads per block
    add<<<10, 10>>>(d_a, d_b, d_c, size);

    // 4. Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < 10; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // 5. Free memory
    delete[] h_a, h_b, h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

### Important Keywords:

* `__global__` = specifier used to denote a kernel function run on GPU
* `cudaMalloc(void *ptr, size_t size)` = allocates memory on device
* `cudaMemcpy(void* dst, void* src, size_t size, cudaMemcpyKind kind)` = copies data between host and device
  * kind = `cudaMemcpyHostToDevice` or `cudaMemcpyDeviceToHost`
* `cudaFree(void* ptr)` = Free Memory
* Thread and block management:
  * `threadIdx.x` - index of thread in block
  * `blockIdx.x` - index of block within grid
  * `blockDim.x` - size of block (threads per block)
  * `gridDim.x` - size of grid (blocks per grid)
