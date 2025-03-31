# Introduction to Massively Parallel

## CPU vs GPU

### CPU:

* **Goal**:

  * must be good at all tasks
  * Minimize latency for 1 thread
* Design Choice:

  * Large on-chip caches
  * Sophisticated control lagic

### GPU

* Goal
  * assume work load is highly parallel
  * maximize throughput for all threads
* Design Choice:
  * multithreading replaces big caches
  * More resources (registers, bandwith, etc) to allow more threading
  * shared control logic across threads

## NVIDIA GPU Architecture

### SIMT execution

* SIMT = Single Instruction Multiple Thread
* Threads run in groups of 32 called Warps
* Threads in a warp share instruction unit (IU)
* Hardware automaticallly handles divergence

### Processing

GPUs consist of many SMs that are all connected via the L2 cache

#### Streaming Multiprocessor (SM)

* fundamental building block of all Nvidia GPUs that contains many cores
* Core Types:
  1. Cuda Cores - General Purpose Executions
  2. Tensor Cores - Optimized for deep learning operations (eg matrix multiplication)
  3. Ray Tracing Cores - Accelerate real time ray tracing for graphics
* Other Components:
  * Warp Schedulers - Manage instruction execution for warps (group of 32 threads)
  * Memory Units - Provides access to registers, caches, and global memory


## Enter CUDA
