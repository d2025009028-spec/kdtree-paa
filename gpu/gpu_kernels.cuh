// Kernels CUDA de busca 1-NN: KD-Tree na GPU e forca bruta na GPU.
// Cada thread da GPU resolve UMA consulta. Usado pelos testes (tests/teste_gpu.cu).
// Compilar com nvcc, por exemplo:
//   nvcc -O3 -arch=sm_89 -std=c++17 -Iinclude src/kdtree.cpp tests/teste_gpu.cu -o teste_gpu.exe
#pragma once
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "kdtree.h"

#ifndef CUDA_OK
#define CUDA_OK(x)                                                         \
    do {                                                                   \
        cudaError_t e_ = (x);                                              \
        if (e_ != cudaSuccess) {                                           \
            printf("ERRO CUDA (%s:%d): %s\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(e_));                                \
            exit(1);                                                       \
        }                                                                  \
    } while (0)
#endif

namespace kdgpu {

using namespace kd;

// ---------------- Kernel: 1-NN na KD-Tree ----------------
// Mesma logica da busca recursiva do Caua (lado "perto" primeiro; o lado "longe"
// so e visitado se o plano de corte estiver mais perto que o melhor ja achado),
// mas com uma pilha explicita. Cada entrada da pilha guarda o no "longe" e a
// distancia ao quadrado ate o plano de corte, para decidir a poda ao desempilhar.
template <typename T>
__global__ void kd_1nn(const No* nos, const T* P, int raiz, int dim, const T* Q,
                       int m, int* melhor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const T* q = Q + (size_t)i * dim;

    int bi = -1;
    T bd = (T)INFINITY;

    int pilha_no[64];
    T pilha_d2[64];
    int sp = 0;

    int no = raiz;
    while (true) {
        // Desce pelo lado "perto" ate o fim do galho.
        while (no != -1) {
            const No x = nos[no];
            const T* p = P + (size_t)x.ponto * dim;
            T d2 = 0;
            for (int k = 0; k < dim; k++) {
                T t = p[k] - q[k];
                d2 += t * t;
            }
            if (d2 < bd) {
                bd = d2;
                bi = x.ponto;
            }
            T dif = q[x.eixo] - p[x.eixo];
            int perto = (dif < 0) ? x.esquerda : x.direita;
            int longe = (dif < 0) ? x.direita : x.esquerda;
            if (longe != -1) {
                pilha_no[sp] = longe;
                pilha_d2[sp] = dif * dif;
                sp++;
            }
            no = perto;
        }
        // Volta: pega o proximo lado "longe" que ainda pode ter algo melhor.
        no = -1;
        while (sp > 0) {
            sp--;
            if (pilha_d2[sp] < bd) {
                no = pilha_no[sp];
                break;
            }
        }
        if (no == -1) break;
    }
    melhor[i] = bi;
}

// ---------------- Kernel: forca bruta ----------------
template <typename T>
__global__ void bf_1nn(const T* P, int n, const T* Q, int m, int dim, int* melhor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const T* q = Q + (size_t)i * dim;
    int bi = 0;
    T bd = 0;
    for (int k = 0; k < dim; k++) {
        T t = P[k] - q[k];
        bd += t * t;
    }
    for (int j = 1; j < n; j++) {
        const T* p = P + (size_t)j * dim;
        T d = 0;
        for (int k = 0; k < dim; k++) {
            T t = p[k] - q[k];
            d += t * t;
        }
        if (d < bd) {
            bd = d;
            bi = j;
        }
    }
    melhor[i] = bi;
}

// Roda a busca 1-NN na GPU (KD-Tree se usa_kd, senao forca bruta) com precisao T
// (double ou float) e devolve, para cada consulta, o indice do ponto mais proximo.
template <typename T>
std::vector<int> busca_gpu_1nn(bool usa_kd, const ArvoreKD& arv,
                               const std::vector<Escalar>& Q, int m) {
    std::vector<int> res(m > 0 ? m : 0, -1);
    if (m <= 0) return res;
    const int n = arv.tamanho(), dim = arv.dimensao(), raiz = arv.raiz();
    const std::vector<No>& nos = arv.nos();

    std::vector<T> hP(arv.pontos().begin(), arv.pontos().end());
    std::vector<T> hQ(Q.begin(), Q.begin() + (size_t)m * dim);

    No* dnos;
    T *dP, *dQ;
    int* dbest;
    CUDA_OK(cudaMalloc(&dnos, nos.size() * sizeof(No)));
    CUDA_OK(cudaMalloc(&dP, hP.size() * sizeof(T)));
    CUDA_OK(cudaMalloc(&dQ, hQ.size() * sizeof(T)));
    CUDA_OK(cudaMalloc(&dbest, m * sizeof(int)));
    CUDA_OK(cudaMemcpy(dnos, nos.data(), nos.size() * sizeof(No), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dP, hP.data(), hP.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(T), cudaMemcpyHostToDevice));

    const int B = 128, G = (m + B - 1) / B;
    if (usa_kd)
        kd_1nn<T><<<G, B>>>(dnos, dP, raiz, dim, dQ, m, dbest);
    else
        bf_1nn<T><<<G, B>>>(dP, n, dQ, m, dim, dbest);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(res.data(), dbest, m * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(dnos);
    cudaFree(dP);
    cudaFree(dQ);
    cudaFree(dbest);
    return res;
}

}  // namespace kdgpu
