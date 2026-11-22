// Testes das versoes em GPU (CUDA): a GPU tem que achar um vizinho tao proximo quanto o da
// KD-Tree sequencial na CPU (empates sao aceitos: o ponto pode ser outro, mas a distancia
// tem que ser a mesma). Cobre KD-Tree e forca bruta, em double e em float.
// Compilar (x64 Native Tools Command Prompt):
//   nvcc -O3 -arch=sm_89 -std=c++17 -Iinclude src/kdtree.cpp tests/teste_gpu.cu -o teste_gpu.exe
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../gpu/gpu_kernels.cuh"
#include "kdtree.h"

using namespace kd;
using namespace kdgpu;

static int falhas = 0;

static void confere(bool ok, const char* nome) {
    printf("  [%s] %s\n", ok ? "OK" : "FALHOU", nome);
    if (!ok) falhas++;
}

static std::vector<Escalar> aleatorios(size_t qtd, unsigned semente) {
    std::mt19937 g(semente);
    std::uniform_real_distribution<Escalar> U(0.0, 1.0);
    std::vector<Escalar> v(qtd);
    for (auto& x : v) x = U(g);
    return v;
}

// Compara a GPU com a KD-Tree da CPU. Recalcula, em double, a distancia do ponto que a GPU
// escolheu e compara com a distancia do vizinho da CPU. Devolve quantas consultas diferem.
template <typename T>
static int compara(const ArvoreKD& arv, const std::vector<Escalar>& Q, int m, bool usa_kd,
                   double tol_rel, double tol_abs) {
    const int n = arv.tamanho(), dim = arv.dimensao();
    std::vector<int> idx = busca_gpu_1nn<T>(usa_kd, arv, Q, m);
    int dif = 0;
    for (int i = 0; i < m; i++) {
        int j = idx[i];
        if (j < 0 || j >= n) { dif++; continue; }  // indice invalido
        double d = 0;
        for (int k = 0; k < dim; k++) {
            double t = arv.pontos()[(size_t)j * dim + k] - Q[(size_t)i * dim + k];
            d += t * t;
        }
        double ref = arv.vizinho_mais_proximo(&Q[(size_t)i * dim]).distancia2;
        if (std::fabs(d - ref) > tol_rel * ref + tol_abs) dif++;
    }
    return dif;
}

int main() {
    cudaDeviceProp p;
    CUDA_OK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s\n", p.name);

    // 1) Dados aleatorios, varias dimensoes.
    for (int dim : {2, 3, 10, 50}) {
        const int n = 3000, m = 500;
        auto P = aleatorios((size_t)n * dim, 1);
        auto Q = aleatorios((size_t)m * dim, 2);
        ArvoreKD arv(P.data(), n, dim);
        printf("d=%d (n=%d, %d consultas)\n", dim, n, m);
        confere(compara<double>(arv, Q, m, true, 1e-12, 1e-15) == 0, "KD-Tree na GPU, double");
        confere(compara<float>(arv, Q, m, true, 1e-3, 1e-6) == 0, "KD-Tree na GPU, float");
        confere(compara<double>(arv, Q, m, false, 1e-12, 1e-15) == 0, "forca bruta na GPU, double");
        confere(compara<float>(arv, Q, m, false, 1e-3, 1e-6) == 0, "forca bruta na GPU, float");
    }

    // 2) Casos de borda.
    printf("casos de borda\n");
    {
        std::vector<Escalar> P = {0.3, 0.7};
        std::vector<Escalar> Q = {0.9, 0.1, 0.0, 0.0};
        ArvoreKD arv(P.data(), 1, 2);
        confere(compara<double>(arv, Q, 2, true, 1e-12, 1e-15) == 0 &&
                    compara<double>(arv, Q, 2, false, 1e-12, 1e-15) == 0,
                "arvore com 1 ponto");
    }
    {
        auto P = aleatorios(300 * 3, 5);
        auto Q = aleatorios(3, 6);
        ArvoreKD arv(P.data(), 300, 3);
        confere(compara<double>(arv, Q, 1, true, 1e-12, 1e-15) == 0, "1 consulta so");
    }
    {
        // lote vazio nao pode dar erro
        auto P = aleatorios(100 * 2, 7);
        ArvoreKD arv(P.data(), 100, 2);
        std::vector<Escalar> Q;
        confere(busca_gpu_1nn<double>(true, arv, Q, 0).empty(), "lote vazio (0 consultas)");
    }
    {
        // todos os pontos iguais: qualquer indice serve, mas a distancia tem que bater
        const int n = 500, dim = 4, m = 200;
        std::vector<Escalar> P((size_t)n * dim, 0.5);
        auto Q = aleatorios((size_t)m * dim, 8);
        ArvoreKD arv(P.data(), n, dim);
        confere(compara<double>(arv, Q, m, true, 1e-12, 1e-15) == 0 &&
                    compara<float>(arv, Q, m, true, 1e-3, 1e-6) == 0,
                "todos os pontos iguais (empates)");
    }
    {
        // consultas iguais a pontos da arvore: distancia exatamente 0
        const int n = 1000, dim = 5, m = 300;
        auto P = aleatorios((size_t)n * dim, 9);
        ArvoreKD arv(P.data(), n, dim);
        std::vector<Escalar> Q(P.begin(), P.begin() + (size_t)m * dim);
        confere(compara<double>(arv, Q, m, true, 0.0, 1e-15) == 0 &&
                    compara<double>(arv, Q, m, false, 0.0, 1e-15) == 0,
                "consulta igual a um ponto da arvore (double): distancia 0");
        confere(compara<float>(arv, Q, m, true, 0.0, 1e-9) == 0,
                "consulta igual a um ponto da arvore (float): distancia ~0");
    }

    printf("\n%s (%d falha%s)\n", falhas == 0 ? "TODOS OS TESTES PASSARAM" : "HOUVE FALHAS",
           falhas, falhas == 1 ? "" : "s");
    return falhas == 0 ? 0 : 1;
}
