
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <stdlib.h>

const int m = 14400;
const int n = 14400;

// CUDA Kernel for MatrixAddition
__global__ void MatAdd(float* A, float* B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculates Current Index
    if (i < m*n)
        C[i] = A[i] + B[i];
}

float* createRandomMatrix(float *matrix, int m, int n) {
    matrix = new float[m * n];
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            matrix[n * r + c] = static_cast <float> (rand() % 10) / 1.0;
        }
    }
    return matrix;
}

float* createEmptyMatrix(float* matrix, int m, int n) {
    matrix = new float[m * n];
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            matrix[n * r + c] = 0.0;
        }
    }
    return matrix;
}

int main() {
    float* A = (float*)malloc(m * n);
    float* B = (float*)malloc(m * n);
    float* C = (float*)malloc(m * n);

    float* d_A;
    float* d_B;
    float* d_C;

    auto start1 = std::chrono::high_resolution_clock::now();
    std::cout << "[+] Generation of Matrices started \n";
    A = createRandomMatrix(A, m, n);
    std::cout << "[+] Generation of Matrix A finished \n";
    B = createRandomMatrix(B, m, n);
    std::cout << "[+] Generation of Matrix B finished \n";
    C = createEmptyMatrix(C, m, n);
    std::cout << "[+] Generation of Matrix C finished \n";
    auto stop1 = std::chrono::high_resolution_clock::now();

    std::cout << "[+] Generation on CPU finished \n[+] Duration: " << std::chrono::duration<double>(stop1 - start1).count() << " seconds\n";

    int blockSize = 64; // Block Size of GPU, 64 for RTX 2070super
    int numBlocks = ((n*m) + blockSize - 1) / blockSize; // Calculates the number of Blocks

    // Allocate Memory on GPU
    cudaMalloc(&d_A, (m * n) * sizeof(float));
    cudaMalloc(&d_B, (m * n) * sizeof(float));
    cudaMalloc(&d_C, (m * n) * sizeof(float));

    // Copy Data to GPU
    cudaMemcpy(d_A, A, (m * n) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, (m * n) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, (m * n) * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "[+] Using " << numBlocks << " Blocks with " << blockSize << " Threads\n";
    std::cout << "[+] Calculation started with " << (numBlocks * blockSize) << " Threads";
    auto start = std::chrono::high_resolution_clock::now();

    // Start Kernel
    MatAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C);

    // Wait for Calculation to finish
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();

    // Copy result to Host
    cudaMemcpy(C, d_C, (m * n) * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "\n[+] Multithreaded calculation finished \n[+] Duration: " << std::chrono::duration<double>(stop - start).count() << " seconds";

    // Free Memory on GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free memory on host
    delete[] A;
    delete[] B;
    delete[] C;
}
