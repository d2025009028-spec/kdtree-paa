// Benchmark FINAL consolidado (itens 4, 5 e 6).
// Para cada dimensao (2, 3, 10, 50) mede, com o MESMO conjunto de consultas e nas mesmas
// condicoes, a busca 1-NN com:
//   KD-Tree CPU sequencial, KD-Tree CPU OpenMP (todas as threads do sistema),
//   KD-Tree na GPU (double e float) e forca bruta na GPU (double e float).
// Cada tempo e a MEDIANA das repeticoes. O sequencial e medido de novo no fim de cada
// dimensao para checar a estabilidade. Grava resultados_final.csv.
// Uso: bench_final.exe [repeticoes_cpu] [escala] [repeticoes_gpu]
//   padrao: 3 repeticoes na CPU, escala 1.0, 7 repeticoes na GPU (os kernels da GPU sao
//   curtos e variam mais entre repeticoes, entao a mediana usa mais amostras).
//   escala multiplica o numero de consultas (use 0.1 para um teste rapido).
//   4o argumento (opcional): nome do CSV de saida (padrao resultados_final.csv).
// Compilar (x64 Native Tools Command Prompt):
//   nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/O2,/openmp" -Iinclude src/kdtree.cpp benchmarks/bench_final.cu -o bench_final.exe
// IMPORTANTE: sem -O3 e /O2 o nvcc nao otimiza o codigo da CPU e os tempos saem piores.
// Os kernels da GPU ficam em gpu/gpu_kernels.cuh (compartilhado com tests/teste_gpu.cu).
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../gpu/gpu_kernels.cuh"  // kernels da GPU (KD-Tree e forca bruta), CUDA_OK
#include "kdtree.h"

using namespace kd;
using namespace kdgpu;

// Relogio de alta resolucao (nao depende da resolucao do omp_get_wtime).
static double agora() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static double mediana(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ---------------- KD-Tree na CPU ----------------
static void kd_seq(const ArvoreKD& a, const std::vector<Escalar>& Q, int m, int dim,
                   std::vector<Vizinho>& out) {
    out.resize(m);
    for (int i = 0; i < m; i++) out[i] = a.vizinho_mais_proximo(&Q[(size_t)i * dim]);
}

static void kd_omp(const ArvoreKD& a, const std::vector<Escalar>& Q, int m, int dim,
                   std::vector<Vizinho>& out, int threads) {
    out.resize(m);
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 64)
    for (int i = 0; i < m; i++) out[i] = a.vizinho_mais_proximo(&Q[(size_t)i * dim]);
}

static double mede_seq(const ArvoreKD& arv, const std::vector<Escalar>& Q, int m, int dim,
                       int reps, std::vector<Vizinho>& out) {
    std::vector<double> ts;
    for (int r = 0; r < reps; r++) {
        double t0 = agora();
        kd_seq(arv, Q, m, dim, out);
        ts.push_back(agora() - t0);
    }
    return mediana(ts);
}

template <typename T>
void lanca(bool usa_kd, const No* dnos, const T* dP, int raiz, const T* dQ, int n,
           int m, int dim, int* dbest, int G, int B) {
    if (usa_kd)
        kd_1nn<T><<<G, B>>>(dnos, dP, raiz, dim, dQ, m, dbest);
    else
        bf_1nn<T><<<G, B>>>(dP, n, dQ, m, dim, dbest);
}

struct ResGPU {
    double kernel, total;
    int dif;
};

// Roda a busca na GPU (KD-Tree ou forca bruta) em precisao T e confere com 'ref'.
template <typename T>
ResGPU roda_gpu(bool usa_kd, const ArvoreKD& arv, const std::vector<Escalar>& Q, int m,
                const std::vector<Vizinho>& ref, int reps) {
    const int n = arv.tamanho(), dim = arv.dimensao(), raiz = arv.raiz();
    const std::vector<Escalar>& P = arv.pontos();
    const std::vector<No>& nos = arv.nos();

    std::vector<T> hP(P.begin(), P.end()), hQ(Q.begin(), Q.end());
    std::vector<int> hbest(m);

    No* dnos;
    T *dP, *dQ;
    int* dbest;
    CUDA_OK(cudaMalloc(&dnos, nos.size() * sizeof(No)));
    CUDA_OK(cudaMalloc(&dP, hP.size() * sizeof(T)));
    CUDA_OK(cudaMalloc(&dQ, hQ.size() * sizeof(T)));
    CUDA_OK(cudaMalloc(&dbest, m * sizeof(int)));
    const int B = 128, G = (m + B - 1) / B;

    auto copia_entrada = [&]() {
        CUDA_OK(cudaMemcpy(dnos, nos.data(), nos.size() * sizeof(No), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dP, hP.data(), hP.size() * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(T), cudaMemcpyHostToDevice));
    };

    // Aquecimento (a primeira chamada na GPU e mais lenta; nao entra na medicao).
    copia_entrada();
    lanca<T>(usa_kd, dnos, dP, raiz, dQ, n, m, dim, dbest, G, B);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    // Mantem a GPU ocupada por ~0,3 s para o clock estabilizar antes de medir
    // (kernels curtos medidos com a placa "fria" variam bastante).
    for (double tw = agora(); agora() - tw < 0.3;) {
        lanca<T>(usa_kd, dnos, dP, raiz, dQ, n, m, dim, dbest, G, B);
        CUDA_OK(cudaDeviceSynchronize());
    }

    std::vector<double> tk, tt;
    cudaEvent_t e0, e1;
    CUDA_OK(cudaEventCreate(&e0));
    CUDA_OK(cudaEventCreate(&e1));
    for (int r = 0; r < reps; r++) {
        // so o kernel
        CUDA_OK(cudaEventRecord(e0));
        lanca<T>(usa_kd, dnos, dP, raiz, dQ, n, m, dim, dbest, G, B);
        CUDA_OK(cudaEventRecord(e1));
        CUDA_OK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
        tk.push_back(ms / 1000.0);

        // total: copias CPU->GPU, kernel e resultado GPU->CPU
        double t0 = agora();
        copia_entrada();
        lanca<T>(usa_kd, dnos, dP, raiz, dQ, n, m, dim, dbest, G, B);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaMemcpy(hbest.data(), dbest, m * sizeof(int), cudaMemcpyDeviceToHost));
        tt.push_back(agora() - t0);
    }

    // Confere contra a KD-Tree da CPU (aceita empates; em float, pequena tolerancia).
    double tol = (sizeof(T) == 4) ? 1e-3 : 1e-12;
    int dif = 0;
    for (int i = 0; i < m; i++) {
        int j = hbest[i];
        if (j < 0) { dif++; continue; }
        double d = 0;
        for (int k = 0; k < dim; k++) {
            double t = P[(size_t)j * dim + k] - Q[(size_t)i * dim + k];
            d += t * t;
        }
        if (std::fabs(d - ref[i].distancia2) > tol * ref[i].distancia2 + 1e-15) dif++;
    }

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(dnos);
    cudaFree(dP);
    cudaFree(dQ);
    cudaFree(dbest);
    return {mediana(tk), mediana(tt), dif};
}

int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 3;
    double escala = argc > 2 ? atof(argv[2]) : 1.0;
    int repsg = argc > 3 ? atoi(argv[3]) : 7;
    const char* arq = argc > 4 ? argv[4] : "resultados_final.csv";
    const int n = 100000;
    const int th = omp_get_max_threads();

    struct Cfg {
        int dim, m;
    };
    const Cfg cfgs[] = {{2, 200000}, {3, 200000}, {10, 20000}, {50, 1000}};

    cudaDeviceProp p;
    CUDA_OK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s | n=%d pontos | %d repeticoes CPU e %d GPU (mediana) | OpenMP=%d threads | escala=%.2f\n",
           p.name, n, reps, repsg, th, escala);

    FILE* csv = fopen(arq, "w");
    if (csv) fprintf(csv, "dim,consultas,metodo,tempo_s,speedup_vs_seq\n");
    auto grava = [&](int dim, int m, const char* metodo, double t, double tseq) {
        if (csv) fprintf(csv, "%d,%d,%s,%.6f,%.4f\n", dim, m, metodo, t, t > 0 ? tseq / t : 0.0);
    };

    for (const Cfg& c : cfgs) {
        int dim = c.dim;
        int m = std::max(100, (int)(c.m * escala));

        std::mt19937 g(42);
        std::uniform_real_distribution<Escalar> U(0.0, 1.0);
        std::vector<Escalar> P((size_t)n * dim), Q((size_t)m * dim);
        for (auto& x : P) x = U(g);
        for (auto& x : Q) x = U(g);
        ArvoreKD arv(P.data(), n, dim);

        std::vector<Vizinho> ref, out, tmp;
        kd_seq(arv, Q, std::min(m, 200), dim, tmp);  // aquecimento (nao medido)

        double tseq = mede_seq(arv, Q, m, dim, reps, ref);

        std::vector<double> to;
        for (int r = 0; r < reps; r++) {
            double t0 = agora();
            kd_omp(arv, Q, m, dim, out, th);
            to.push_back(agora() - t0);
        }
        double tomp = mediana(to);
        int difo = 0;
        for (int i = 0; i < m; i++)
            if (out[i].distancia2 != ref[i].distancia2) difo++;

        ResGPU kd_d = roda_gpu<double>(true, arv, Q, m, ref, repsg);
        ResGPU kd_f = roda_gpu<float>(true, arv, Q, m, ref, repsg);
        ResGPU bf_d = roda_gpu<double>(false, arv, Q, m, ref, repsg);
        ResGPU bf_f = roda_gpu<float>(false, arv, Q, m, ref, repsg);

        double tseq2 = mede_seq(arv, Q, m, dim, reps, tmp);

        printf("\n== d=%d (%d consultas) ==\n", dim, m);
        printf("KD-Tree CPU sequencial      : %.4f s (de novo no fim: %.4f s, %+.0f%%)\n", tseq,
               tseq2, 100.0 * (tseq2 - tseq) / tseq);
        printf("KD-Tree CPU OpenMP %2d thr.  : %.4f s | speedup %.2f | diferencas %d\n", th, tomp,
               tseq / tomp, difo);
        printf("GPU KD-Tree    double: kernel %.4f s | total %.4f s | speedup %.2f | dif %d\n",
               kd_d.kernel, kd_d.total, tseq / kd_d.total, kd_d.dif);
        printf("GPU KD-Tree    float : kernel %.4f s | total %.4f s | speedup %.2f | dif %d\n",
               kd_f.kernel, kd_f.total, tseq / kd_f.total, kd_f.dif);
        printf("GPU forca bruta double: kernel %.4f s | total %.4f s | speedup %.2f | dif %d\n",
               bf_d.kernel, bf_d.total, tseq / bf_d.total, bf_d.dif);
        printf("GPU forca bruta float : kernel %.4f s | total %.4f s | speedup %.2f | dif %d\n",
               bf_f.kernel, bf_f.total, tseq / bf_f.total, bf_f.dif);
        fflush(stdout);

        char nome[64];
        grava(dim, m, "kdtree_seq", tseq, tseq);
        grava(dim, m, "kdtree_seq_fim", tseq2, tseq);
        snprintf(nome, sizeof nome, "kdtree_omp%d", th);
        grava(dim, m, nome, tomp, tseq);
        grava(dim, m, "gpu_kdtree_double_kernel", kd_d.kernel, tseq);
        grava(dim, m, "gpu_kdtree_double_total", kd_d.total, tseq);
        grava(dim, m, "gpu_kdtree_float_kernel", kd_f.kernel, tseq);
        grava(dim, m, "gpu_kdtree_float_total", kd_f.total, tseq);
        grava(dim, m, "gpu_bf_double_kernel", bf_d.kernel, tseq);
        grava(dim, m, "gpu_bf_double_total", bf_d.total, tseq);
        grava(dim, m, "gpu_bf_float_kernel", bf_f.kernel, tseq);
        grava(dim, m, "gpu_bf_float_total", bf_f.total, tseq);
    }
    if (csv) fclose(csv);
    printf("\nResultados gravados em %s\n", arq);
    return 0;
}
