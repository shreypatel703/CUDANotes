#include <iostream>
#include <cuda_runtime.h>


__global__ void addKernel(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void solve(const float *A, const float *B, float *C, int N) {

    // 1. Allocate memory on device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    //2. Copy data from host to device
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    //3. Launch kernel
    blocksPerGrid = (N + 256 - 1) / 256;
    threadsPerBlock = 256;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    //4. Copy data from device to host
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    //5. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

    

int main() {
    const int N = 1 << 20; // 1 million elements
    // Initialize input vectors
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i;
    }
    
    // Solve the problem
    solve(h_A, h_B, h_C, N);

    // Print the result
    for (int i = 0; i < N; i++) {
        std::cout << h_C[i] << " ";
    }

}
