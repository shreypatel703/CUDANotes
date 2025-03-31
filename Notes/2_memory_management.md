## **Day 2: CUDA Memory Management & Large-Scale Computation**  
### **Topics to Learn**  
#### **1. CUDA Memory Hierarchy**  
   - **Global memory** (high latency, accessible to all threads)  
   - **Shared memory** (fast but limited, per thread block)  
   - **Local memory** (per-thread private memory)  
   - **Constant memory** (optimized for read-heavy operations)  

#### **2. CUDA Memory Allocation & Transfers**  
   - `cudaMalloc()` & `cudaMemcpy()` for memory allocation and data transfer  
   - **Pinned (page-locked) memory** for faster CPU-GPU transfers  

#### **3. Measuring Performance**  
   - Using **CUDA events** to measure execution time  

### **Project: Large Vector Addition with Memory Transfers**  
- Extend Day 1’s project to **millions of elements**.  
- Compare **CPU vs. GPU execution time**.  