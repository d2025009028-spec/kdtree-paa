// Consultas em lote na KD-Tree: versao sequencial e versao paralela (OpenMP).
// Cada consulta e independente: cada thread escreve so na sua posicao da saida.
// Compile com -fopenmp (sem a opcao, os #pragma sao ignorados e o codigo roda sequencial).
#pragma once
#include <vector>

#include "kdtree.h"

namespace kd {

// Q tem m consultas de 'dim' valores cada (mesmo formato dos pontos da arvore).
inline void lote_1nn_seq(const ArvoreKD& a, const std::vector<Escalar>& Q, int m,
                         std::vector<Vizinho>& saida) {
    int dim = a.dimensao();
    saida.resize(m);
    for (int i = 0; i < m; i++)
        saida[i] = a.vizinho_mais_proximo(&Q[(size_t)i * dim]);
}

inline void lote_1nn_omp(const ArvoreKD& a, const std::vector<Escalar>& Q, int m,
                         std::vector<Vizinho>& saida, int threads) {
    int dim = a.dimensao();
    saida.resize(m);
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 64)
    for (int i = 0; i < m; i++)
        saida[i] = a.vizinho_mais_proximo(&Q[(size_t)i * dim]);
}

inline void lote_knn_seq(const ArvoreKD& a, const std::vector<Escalar>& Q, int m, int k,
                         std::vector<std::vector<Vizinho>>& saida) {
    int dim = a.dimensao();
    saida.resize(m);
    for (int i = 0; i < m; i++)
        saida[i] = a.k_vizinhos(&Q[(size_t)i * dim], k);
}

inline void lote_knn_omp(const ArvoreKD& a, const std::vector<Escalar>& Q, int m, int k,
                         std::vector<std::vector<Vizinho>>& saida, int threads) {
    int dim = a.dimensao();
    saida.resize(m);
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 64)
    for (int i = 0; i < m; i++)
        saida[i] = a.k_vizinhos(&Q[(size_t)i * dim], k);
}

}  // namespace kd
