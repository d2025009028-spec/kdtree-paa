// Item 6: speedup do OpenMP em funcao do numero de threads, para cada dimensao.
// Mede o 1-NN em lote com a KD-Tree: sequencial vs OpenMP com varios numeros de threads.
// Cada tempo e a MEDIANA das repeticoes. Grava resultados_threads.csv.
// Uso: bench_threads.exe [repeticoes] [escala]
//   repeticoes: padrao 3.  escala: multiplica o numero de consultas (padrao 1.0;
//   use 0.1 para um teste rapido).
// Para checar a estabilidade da medicao, o tempo sequencial e medido duas vezes
// (no inicio e no fim de cada dimensao). Dois speedups sao mostrados: contra a versao
// sequencial e contra o proprio OpenMP com 1 thread (escalonamento "puro").
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "kdtree.h"

using namespace kd;

static double mediana(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static void lote_seq(const ArvoreKD& a, const std::vector<Escalar>& Q, int m, int dim,
                     std::vector<Vizinho>& out) {
    out.resize(m);
    for (int i = 0; i < m; i++) out[i] = a.vizinho_mais_proximo(&Q[(size_t)i * dim]);
}

static void lote_omp(const ArvoreKD& a, const std::vector<Escalar>& Q, int m, int dim,
                     std::vector<Vizinho>& out, int threads) {
    out.resize(m);
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 64)
    for (int i = 0; i < m; i++) out[i] = a.vizinho_mais_proximo(&Q[(size_t)i * dim]);
}

static double mede_seq(const ArvoreKD& arv, const std::vector<Escalar>& Q, int m, int dim,
                       int reps, std::vector<Vizinho>& out) {
    std::vector<double> ts;
    for (int r = 0; r < reps; r++) {
        double t0 = omp_get_wtime();
        lote_seq(arv, Q, m, dim, out);
        ts.push_back(omp_get_wtime() - t0);
    }
    return mediana(ts);
}

int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 3;
    double escala = argc > 2 ? atof(argv[2]) : 1.0;

    const int n = 100000;
    struct Cfg {
        int dim, m;
    };
    const Cfg cfgs[] = {{2, 200000}, {3, 200000}, {10, 20000}, {50, 1000}};
    const int threads[] = {1, 2, 4, 6, 8, 12, 14, 16, 20};

    printf("n=%d pontos, %d repeticoes (mediana), escala de consultas=%.2f, "
           "maximo de threads do sistema=%d\n",
           n, reps, escala, omp_get_max_threads());

    FILE* csv = fopen("resultados_threads.csv", "w");
    if (csv) fprintf(csv, "dim,consultas,metodo,threads,tempo_s,speedup_vs_seq,speedup_vs_1thread\n");

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
        lote_seq(arv, Q, std::min(m, 200), dim, tmp);  // aquecimento (nao medido)

        double tseq = mede_seq(arv, Q, m, dim, reps, ref);

        printf("\n== d=%d (%d consultas) ==\n", dim, m);
        printf("sequencial (inicio): %.4f s\n", tseq);
        printf("threads | tempo (s) | vs seq | vs 1 thread | eficiencia | diferencas\n");
        if (csv) fprintf(csv, "%d,%d,seq_inicio,0,%.6f,1.0000,\n", dim, m, tseq);

        double t1 = 0;
        for (int th : threads) {
            std::vector<double> to;
            for (int r = 0; r < reps; r++) {
                double t0 = omp_get_wtime();
                lote_omp(arv, Q, m, dim, out, th);
                to.push_back(omp_get_wtime() - t0);
            }
            double t = mediana(to);
            if (th == 1) t1 = t;
            int dif = 0;
            for (int i = 0; i < m; i++)
                if (out[i].distancia2 != ref[i].distancia2) dif++;
            double sp_seq = tseq / t, sp_1 = t1 / t;
            printf("%7d | %9.4f | %6.2f | %11.2f | %9.0f%% | %d\n", th, t, sp_seq, sp_1,
                   100.0 * sp_1 / th, dif);
            fflush(stdout);
            if (csv) fprintf(csv, "%d,%d,omp,%d,%.6f,%.4f,%.4f\n", dim, m, th, t, sp_seq, sp_1);
        }

        double tseq2 = mede_seq(arv, Q, m, dim, reps, ref);
        printf("sequencial (fim)   : %.4f s  (diferenca para o inicio: %+.0f%%)\n", tseq2,
               100.0 * (tseq2 - tseq) / tseq);
        if (csv) fprintf(csv, "%d,%d,seq_fim,0,%.6f,%.4f,\n", dim, m, tseq2, tseq / tseq2);
    }
    if (csv) fclose(csv);
    printf("\nResultados gravados em resultados_threads.csv\n");
    return 0;
}
