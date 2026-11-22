// Testes das versoes paralelas em CPU (OpenMP): o resultado tem que ser IDENTICO ao da
// versao sequencial (mesmo ponto e mesma distancia, para cada consulta), com qualquer
// numero de threads.
// Compilar: g++ -O2 -std=c++17 -fopenmp -Iinclude src/kdtree.cpp tests/teste_paralelo.cpp -o teste_paralelo.exe
#include <cstdio>
#include <random>
#include <vector>

#include "../paralelo/omp_lote.h"
#include "kdtree.h"

using namespace kd;

static int falhas = 0;

static void confere(bool ok, const char* nome) {
    printf("  [%s] %s\n", ok ? "OK" : "FALHOU", nome);
    if (!ok) falhas++;
}

static bool iguais_1nn(const std::vector<Vizinho>& a, const std::vector<Vizinho>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i].indice != b[i].indice || a[i].distancia2 != b[i].distancia2) return false;
    return true;
}

static bool iguais_knn(const std::vector<std::vector<Vizinho>>& a,
                       const std::vector<std::vector<Vizinho>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].size() != b[i].size()) return false;
        for (size_t j = 0; j < a[i].size(); j++)
            if (a[i][j].indice != b[i][j].indice ||
                a[i][j].distancia2 != b[i][j].distancia2)
                return false;
    }
    return true;
}

static std::vector<Escalar> aleatorios(size_t qtd, unsigned semente) {
    std::mt19937 g(semente);
    std::uniform_real_distribution<Escalar> U(0.0, 1.0);
    std::vector<Escalar> v(qtd);
    for (auto& x : v) x = U(g);
    return v;
}

int main() {
    const int threads[] = {1, 2, 3, 4, 8, 16};

    // 1) Dados aleatorios, varias dimensoes, varios numeros de threads.
    for (int dim : {2, 3, 10, 50}) {
        const int n = 3000, m = 700;
        auto P = aleatorios((size_t)n * dim, 1);
        auto Q = aleatorios((size_t)m * dim, 2);
        ArvoreKD arv(P.data(), n, dim);

        std::vector<Vizinho> ref1, out1;
        lote_1nn_seq(arv, Q, m, ref1);
        std::vector<std::vector<Vizinho>> refk, outk;

        printf("d=%d (n=%d, %d consultas)\n", dim, n, m);
        bool ok1 = true;
        for (int th : threads) {
            lote_1nn_omp(arv, Q, m, out1, th);
            ok1 = ok1 && iguais_1nn(ref1, out1);
        }
        confere(ok1, "1-NN OpenMP == sequencial (1, 2, 3, 4, 8 e 16 threads)");

        for (int k : {1, 5, 20}) {
            lote_knn_seq(arv, Q, m, k, refk);
            bool okk = true;
            for (int th : threads) {
                lote_knn_omp(arv, Q, m, k, outk, th);
                okk = okk && iguais_knn(refk, outk);
            }
            char nome[96];
            snprintf(nome, sizeof nome, "k-NN OpenMP == sequencial (k=%d)", k);
            confere(okk, nome);
        }
    }

    // 2) Casos de borda.
    printf("casos de borda\n");
    {
        // arvore com 1 ponto
        std::vector<Escalar> P = {0.3, 0.7};
        std::vector<Escalar> Q = {0.9, 0.1, 0.0, 0.0};
        ArvoreKD arv(P.data(), 1, 2);
        std::vector<Vizinho> s, o;
        lote_1nn_seq(arv, Q, 2, s);
        lote_1nn_omp(arv, Q, 2, o, 4);
        confere(iguais_1nn(s, o) && o[0].indice == 0 && o[1].indice == 0,
                "arvore com 1 ponto: todas as consultas retornam esse ponto");
    }
    {
        // uma unica consulta com mais threads do que consultas
        auto P = aleatorios(300 * 3, 5);
        auto Q = aleatorios(3, 6);
        ArvoreKD arv(P.data(), 300, 3);
        std::vector<Vizinho> s, o;
        lote_1nn_seq(arv, Q, 1, s);
        lote_1nn_omp(arv, Q, 1, o, 16);
        confere(iguais_1nn(s, o), "1 consulta com 16 threads");
    }
    {
        // lote vazio
        auto P = aleatorios(100 * 2, 7);
        ArvoreKD arv(P.data(), 100, 2);
        std::vector<Escalar> Q;
        std::vector<Vizinho> o;
        lote_1nn_omp(arv, Q, 0, o, 8);
        confere(o.empty(), "lote vazio (0 consultas)");
    }
    {
        // pontos repetidos (todas as distancias iguais a 0 ou empatadas)
        const int n = 500, dim = 4, m = 200;
        std::vector<Escalar> P((size_t)n * dim, 0.5);
        auto Q = aleatorios((size_t)m * dim, 8);
        ArvoreKD arv(P.data(), n, dim);
        std::vector<Vizinho> s, o;
        lote_1nn_seq(arv, Q, m, s);
        lote_1nn_omp(arv, Q, m, o, 8);
        confere(iguais_1nn(s, o), "todos os pontos iguais (empates)");
    }
    {
        // consultas que sao copias exatas de pontos da arvore: distancia 0
        const int n = 1000, dim = 5;
        auto P = aleatorios((size_t)n * dim, 9);
        ArvoreKD arv(P.data(), n, dim);
        std::vector<Escalar> Q(P.begin(), P.begin() + (size_t)300 * dim);
        std::vector<Vizinho> o;
        lote_1nn_omp(arv, Q, 300, o, 8);
        bool ok = true;
        for (const auto& v : o) ok = ok && v.distancia2 == 0.0;
        confere(ok, "consulta igual a um ponto da arvore: distancia 0");
    }
    {
        // k maior que o numero de pontos
        auto P = aleatorios(3 * 2, 10);
        auto Q = aleatorios(10 * 2, 11);
        ArvoreKD arv(P.data(), 3, 2);
        std::vector<std::vector<Vizinho>> s, o;
        lote_knn_seq(arv, Q, 10, 10, s);
        lote_knn_omp(arv, Q, 10, 10, o, 4);
        confere(iguais_knn(s, o), "k maior que n: paralelo == sequencial");
    }

    printf("\n%s (%d falha%s)\n", falhas == 0 ? "TODOS OS TESTES PASSARAM" : "HOUVE FALHAS",
           falhas, falhas == 1 ? "" : "s");
    return falhas == 0 ? 0 : 1;
}
