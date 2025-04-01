# Intro to CUDA

## What is CUDA

CUDA (Compute Unified Device Architecture) is a parallel computing platform and API developed by **NVIDIA** that enables developers to leverage the **massively parallel** computing power of **GPUs**

CUDA follows a **heterogeneous computing** model, where a program runs on both:

1. **CPU (Host)** – Handles sequential and control-flow logic.
2. **GPU (Device)** – Executes massively parallel workloads.

Kernels - functions that run in parallel across many threads on the GPU.

#### **Key Components of CUDA**

1. **Threads** : The smallest execution unit in CUDA. Thousands of threads run in parallel.
2. **Warps**: Group of threads inside of a block (32 threads for NVIDIA GPUs) that execute SIMULTANEOUSLY  on the GPU.

   - GPU schedules and executes warps not individual threads
   - All threads in a warp execute the same instruction at a time (SIMT execution)
   - Warp Divergence: If threads within a warp take different execution paths (e.g., an if-else branch), they serialize, reducing efficiency.
3. **Thread Blocks** : A group of threads that execute the same kernel function.

   * A **block** consists of multiple **warps**.
   * The number of threads per block is usually **a multiple of 32** (e.g., 128, 256, or 512).
   * All threads in a block share **shared memory** , which is much faster than global memory
4. **Grid** : A collection of thread blocks. The grid organizes how threads are mapped to the GPU. Each kernel creates its on grid

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

#### CUDA Event Keywords
`cudaEventCreate(cudaEvent_t *event)` - Creates an event object
`cudaEventRecord(cudaEvent_t event)` - Records the event at a specific time
`cudaEventSynchronize(cudaEvent_t event)` - Blocks CPU until the event completes
`cudaEventElapsedTime(float *milliseconds, cudaEvent_t event1, cudaEvent_t event2)` - Computes elapsed time (in ms) between two events
`cudaEventDestroy(cudaEvent_t event)` - Frees memory allocated for the event