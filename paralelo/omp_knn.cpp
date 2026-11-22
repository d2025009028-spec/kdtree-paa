// Item 4 (CPU): benchmark das consultas em lote (1-NN e k-NN), sequencial vs OpenMP.
// As funcoes de lote estao em omp_lote.h.
// Compilar: g++ -O2 -std=c++17 -fopenmp -Iinclude src/kdtree.cpp paralelo/omp_knn.cpp -o omp_knn.exe
#include <omp.h>

#include <iostream>
#include <random>
#include <vector>

#include "kdtree.h"
#include "omp_lote.h"

using namespace kd;

static bool iguais(const std::vector<std::vector<Vizinho>>& a,
                   const std::vector<std::vector<Vizinho>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].size() != b[i].size()) return false;
        for (size_t j = 0; j < a[i].size(); j++)
            if (a[i][j].distancia2 != b[i][j].distancia2) return false;
    }
    return true;
}

int main() {
    const int n = 200000, m = 100000, dim = 3, k = 5;
    std::mt19937 g(42);
    std::uniform_real_distribution<Escalar> U(0.0, 1.0);
    std::vector<Escalar> P((size_t)n * dim), Q((size_t)m * dim);
    for (auto& x : P) x = U(g);
    for (auto& x : Q) x = U(g);

    ArvoreKD arv(P.data(), n, dim);
    std::cout << "n=" << n << " consultas=" << m << " d=" << dim << "\n\n";

    // ---- 1-NN ----
    std::vector<Vizinho> ref, out;
    double t0 = omp_get_wtime();
    lote_1nn_seq(arv, Q, m, ref);
    double tseq = omp_get_wtime() - t0;
    std::cout << "[1-NN] sequencial: " << tseq << " s\n";
    for (int th : {1, 2, 4, 8, 16}) {
        t0 = omp_get_wtime();
        lote_1nn_omp(arv, Q, m, out, th);
        double t = omp_get_wtime() - t0;
        int dif = 0;
        for (int i = 0; i < m; i++)
            if (out[i].distancia2 != ref[i].distancia2) dif++;
        std::cout << "[1-NN] omp " << th << " threads: " << t << " s, speedup "
                  << tseq / t << ", diferencas " << dif << "\n";
    }

    // ---- k-NN ----
    std::vector<std::vector<Vizinho>> refk, outk;
    t0 = omp_get_wtime();
    lote_knn_seq(arv, Q, m, k, refk);
    tseq = omp_get_wtime() - t0;
    std::cout << "\n[" << k << "-NN] sequencial: " << tseq << " s\n";
    for (int th : {1, 2, 4, 8, 16}) {
        t0 = omp_get_wtime();
        lote_knn_omp(arv, Q, m, k, outk, th);
        double t = omp_get_wtime() - t0;
        std::cout << "[" << k << "-NN] omp " << th << " threads: " << t
                  << " s, speedup " << tseq / t << ", resultado igual: "
                  << (iguais(refk, outk) ? "sim" : "NAO") << "\n";
    }
}
