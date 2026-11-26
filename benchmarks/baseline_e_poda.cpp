// Baseline sequencial (forca bruta) x KD-Tree sequencial, e contagem de nos visitados.
// Usa a MESMA configuracao do bench_final: n = 100000 pontos uniformes em [0,1]^d, semente 42
// (pontos primeiro, consultas depois), d = 2, 3, 10, 50, busca 1-NN, tipo double.
//
// Para cada d mede, em uma unica thread e com steady_clock:
//   - tempo por consulta da KD-Tree sequencial (ArvoreKD::vizinho_mais_proximo)
//   - tempo por consulta da forca bruta sequencial (vizinho_mais_proximo_forca_bruta, do Stricker)
//   - quantos nos a busca da KD-Tree visita por consulta (contador numa copia externa da busca,
//     que usa a arvore REAL do projeto; o resultado e conferido contra a biblioteca)
// Como o custo por consulta nao depende de quantas consultas se faz, mede-se uma amostra das
// primeiras consultas (as mesmas do bench_final) em vez das 200 mil, para nao demorar.
//
// Compilar (na raiz do projeto):
//   g++ -O2 -std=c++17 -Iinclude src/kdtree.cpp benchmarks/baseline_e_poda.cpp -o baseline_e_poda.exe
// Uso: baseline_e_poda.exe [escala] [repeticoes] [saida.csv]
//   escala multiplica o tamanho da amostra de consultas (padrao 1.0); repeticoes padrao 3.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "forca_bruta.hpp"
#include "kdtree.h"

using namespace kd;

static double agora() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static double mediana(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Copia da busca 1-NN da biblioteca (mesma logica e mesma poda), so com um contador de nos.
static void buscar_contando(const ArvoreKD& a, int no, const Escalar* q, Vizinho& melhor,
                            long long& visitados) {
    if (no == -1) return;
    visitados++;
    const No& x = a.nos()[no];
    const Escalar* p = &a.pontos()[(size_t)x.ponto * a.dimensao()];
    Escalar d2 = distancia2(p, q, a.dimensao());
    if (d2 < melhor.distancia2) { melhor.indice = x.ponto; melhor.distancia2 = d2; }
    Escalar dif = q[x.eixo] - p[x.eixo];
    int perto = (dif < 0) ? x.esquerda : x.direita;
    int longe = (dif < 0) ? x.direita : x.esquerda;
    buscar_contando(a, perto, q, melhor, visitados);
    if (dif * dif < melhor.distancia2) buscar_contando(a, longe, q, melhor, visitados);
}

int main(int argc, char** argv) {
    double escala = argc > 1 ? atof(argv[1]) : 1.0;
    int reps = argc > 2 ? atoi(argv[2]) : 3;
    const char* arq = argc > 3 ? argv[3] : "baseline_e_poda.csv";
    const int n = 100000;

    struct Cfg { int dim, m_bench, amostra; };
    // m_bench: consultas do bench_final; amostra: quantas consultas medimos aqui.
    const Cfg cfgs[] = {{2, 200000, 5000}, {3, 200000, 5000}, {10, 20000, 2000}, {50, 1000, 200}};

    FILE* csv = fopen(arq, "w");
    if (csv) fprintf(csv, "dim,n,consultas_amostra,kd_seq_us_por_consulta,bf_seq_us_por_consulta,"
                          "bf_sobre_kd,nos_visitados_mediana,fracao_nos_mediana,fracao_nos_max,"
                          "divergencias\n");
    printf("n=%d pontos | %d repeticoes (mediana) | escala da amostra=%.2f | 1 thread\n", n, reps, escala);
    printf("%-4s %-9s %-14s %-14s %-12s %-16s %-10s %-6s\n", "d", "consultas", "KD seq us/q",
           "forca bruta", "bruta/KD", "nos visitados", "fracao", "dif");

    for (const Cfg& c : cfgs) {
        int dim = c.dim;
        int mq = std::max(50, (int)(c.amostra * escala));
        int m_total = std::max(mq, c.m_bench);  // gera o mesmo fluxo de numeros do bench_final

        std::mt19937 g(42);
        std::uniform_real_distribution<Escalar> U(0.0, 1.0);
        std::vector<Escalar> P((size_t)n * dim), Q((size_t)m_total * dim);
        for (auto& x : P) x = U(g);
        for (auto& x : Q) x = U(g);
        ArvoreKD arv(P.data(), n, dim);

        // aquecimento
        for (int i = 0; i < std::min(mq, 50); i++) arv.vizinho_mais_proximo(&Q[(size_t)i * dim]);

        std::vector<Vizinho> rkd(mq), rbf(mq);
        std::vector<double> tkd, tbf;
        for (int r = 0; r < reps; r++) {
            double t0 = agora();
            for (int i = 0; i < mq; i++) rkd[i] = arv.vizinho_mais_proximo(&Q[(size_t)i * dim]);
            tkd.push_back(agora() - t0);
            t0 = agora();
            for (int i = 0; i < mq; i++)
                rbf[i] = vizinho_mais_proximo_forca_bruta(P.data(), n, dim, &Q[(size_t)i * dim]);
            tbf.push_back(agora() - t0);
        }
        int dif = 0;
        for (int i = 0; i < mq; i++)
            if (std::fabs(rkd[i].distancia2 - rbf[i].distancia2) > 1e-12 * (1.0 + rbf[i].distancia2)) dif++;

        // nos visitados por consulta (e confere contra a biblioteca)
        std::vector<double> vis(mq);
        for (int i = 0; i < mq; i++) {
            Vizinho melhor{-1, INFINITY};
            long long v = 0;
            buscar_contando(arv, arv.raiz(), &Q[(size_t)i * dim], melhor, v);
            vis[i] = (double)v;
            if (melhor.distancia2 != rkd[i].distancia2) dif++;
        }
        double vmed = mediana(vis), vmax = *std::max_element(vis.begin(), vis.end());

        double us_kd = mediana(tkd) / mq * 1e6, us_bf = mediana(tbf) / mq * 1e6;
        printf("%-4d %-9d %-14.3f %-14.3f %-12.2f %-16.0f %-9.2f%% %-6d\n", dim, mq, us_kd, us_bf,
               us_bf / us_kd, vmed, 100.0 * vmed / n, dif);
        fflush(stdout);
        if (csv)
            fprintf(csv, "%d,%d,%d,%.4f,%.4f,%.4f,%.0f,%.6f,%.6f,%d\n", dim, n, mq, us_kd, us_bf,
                    us_bf / us_kd, vmed, vmed / n, vmax / n, dif);
    }
    if (csv) fclose(csv);
    printf("\nResultados gravados em %s\n", arq);
    return 0;
}
