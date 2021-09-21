
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <chrono>


using namespace std;
int N = 2048;
int grid_num = 16;
int thread_num = 128;

struct mat4 {
    int* m11, * m12, * m21, * m22;
};

__global__ void mul(int* A, int* B, int* C, int n) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[k * n + i] += A[k * n + j] * B[j * n + i];
        }
    }
}


__global__ void cudaAdd(int* A, int* B, int* C, int n) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < n; i++) {
        C[k * n + i] = A[k * n + i] + B[k * n + i];
    }
}


__global__ void cudaSub(int* A, int* B, int* C, int n) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < n; i++) {
        C[k * n + i] = A[k * n + i] - B[k * n + i];
    }
}


void add(int* A, int* B, int* C) {
    for (int i = 0; i < N * N; i++) {
        C[i] = A[i] + B[i];
    }
    /*
    int* dev_A, * dev_B, * dev_C;

    dev_A = new int[N * N];
    dev_B = new int[N * N];
    dev_C = new int[N * N];

    cudaMalloc((void**)&dev_A, (N * N) * sizeof(int));
    cudaMalloc((void**)&dev_B, (N * N) * sizeof(int));
    cudaMalloc((void**)&dev_C, (N * N) * sizeof(int));


    cudaMemcpy(dev_A, A, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, C, (N * N) * sizeof(int), cudaMemcpyHostToDevice);

    cudaAdd << < grid_num, thread_num >> > (dev_A, dev_B, dev_C, N);
    cudaThreadSynchronize();

    cudaMemcpy(C, dev_C, (N * N) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);*/
}

void sub(int* A, int* B, int* C) {
    for (int i = 0; i < N * N; i++) {
        C[i] = A[i] - B[i];
    }
    
    /*int* dev_A, * dev_B, * dev_C;

    dev_A = new int[N * N];
    dev_B = new int[N * N];
    dev_C = new int[N * N];

    cudaMalloc((void**)&dev_A, (N * N) * sizeof(int));
    cudaMalloc((void**)&dev_B, (N * N) * sizeof(int));
    cudaMalloc((void**)&dev_C, (N * N) * sizeof(int));


    cudaMemcpy(dev_A, A, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, C, (N * N) * sizeof(int), cudaMemcpyHostToDevice);

    cudaSub << < grid_num, thread_num >> > (dev_A, dev_B, dev_C, N);
    cudaThreadSynchronize();

    cudaMemcpy(C, dev_C, (N * N) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);*/
}

void strassen(int* A, int* B, int* C) {
    int* dev_A, * dev_B, * dev_C;

    dev_A = new int[N * N];
    dev_B = new int[N * N];
    dev_C = new int[N * N];

    cudaMalloc((void**)&dev_A, (N * N) * sizeof(int));
    cudaMalloc((void**)&dev_B, (N * N) * sizeof(int));
    cudaMalloc((void**)&dev_C, (N * N) * sizeof(int));

    cudaMemcpy(dev_A, A, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, (N * N) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, C, (N * N) * sizeof(int), cudaMemcpyHostToDevice);

    mul << < grid_num, thread_num >> > (dev_A, dev_B, dev_C, N);
    cudaThreadSynchronize();

    cudaMemcpy(C, dev_C, (N * N) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_A); cudaFree(dev_B); cudaFree(dev_C);
}


int main() {
    // ruin everything
    int* A11, * A12, * A21, * A22,
        * B11, * B12, * B21, * B22,
        * C11, * C12, * C21, * C22,
        * p1, * p2, * p3, * p4, * p5, * p6, * p7,
        * h1, * h2;

    p1 = new int[N * N]; p2 = new int[N * N]; p3 = new int[N * N];
    p4 = new int[N * N]; p5 = new int[N * N]; p6 = new int[N * N];
    p7 = new int[N * N];

    h1 = new int[N * N]; h2 = new int[N * N];


    A11 = new int[N * N]; A12 = new int[N * N];
    A21 = new int[N * N]; A22 = new int[N * N];

    B11 = new int[N * N]; B12 = new int[N * N];
    B21 = new int[N * N]; B22 = new int[N * N];

    C11 = new int[N * N]; C12 = new int[N * N];
    C21 = new int[N * N]; C22 = new int[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A11[i + j * N] = 1; A12[i + j * N] = 1;
            A21[i + j * N] = 1; A22[i + j * N] = 1;

            B11[i + j * N] = 1; B12[i + j * N] = 1;
            B21[i + j * N] = 1; B22[i + j * N] = 1;

            C11[i + j * N] = 0; C12[i + j * N] = 0;
            C21[i + j * N] = 0; C22[i + j * N] = 0;

            p1[i + j * N] = 0; p2[i + j * N] = 0;
            p3[i + j * N] = 0; p4[i + j * N] = 0;
            p5[i + j * N] = 0; p6[i + j * N] = 0;
            p7[i + j * N] = 0;
        }
    }
    //A11[0] = 2; A12[0] = 0; A21[0] = 3; A22[0] = 4;
    //B11[0] = 1; B12[0] = 5; B21[0] = 0; B22[0] = 4;


    auto start = chrono::steady_clock::now();
    cout << "start" << endl;

    add(A11, A22, h1);
    add(B11, B22, h2);
    strassen(h1, h2, p1);

    add(A21, A22, h1);
    strassen(h1, B11, p2);

    sub(B12, B22, h1);
    strassen(A11, h1, p3);

    sub(B21, B11, h1);
    strassen(A22, h1, p4);

    add(A11, A12, h1);
    strassen(h1, B22, p5);

    sub(A21, A11, h1);
    add(B11, B12, h2);
    strassen(h1, h2, p6);

    sub(A12, A22, h1);
    add(B21, B22, h2);
    strassen(h1, h2, p7);

    for (int i = 0; i < N * N; i++) {
        C11[i] = p1[i] + p4[i] - p5[i] + p7[i];
        C12[i] = p3[i] + p5[i];
        C21[i] = p2[i] + p4[i];
        C11[i] = p1[i] - p2[i] + p3[i] + p6[i];
    }

    auto end = chrono::steady_clock::now();

    cout << "Elapsed time in miliseconds : "
        << chrono::duration_cast<chrono::milliseconds>(end - start).count()
        << " ms" << endl;

    int i, j; printf("C11 = \n");
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            printf("%d ", C11[j + i * N]);
        }
        printf("\n");
    }
    cout << C11[0] << endl;

    printf("\n");

    return 0;
}
