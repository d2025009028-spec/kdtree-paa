// Testes de corretude: ArvoreKD (do grupo) vs forca bruta + tempos do baseline.
#include "kdtree.h"
#include "forca_bruta.hpp"
#include "gerador.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace kd;
using Relogio = std::chrono::steady_clock;

static double ms(Relogio::time_point a, Relogio::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static int falhas = 0;
#define VERIFICA(cond, ...) do { if (!(cond)) { ++falhas; std::printf("  FALHA: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static bool perto(Escalar a, Escalar b) { return std::fabs(a - b) <= 1e-12 * (1.0 + std::fabs(b)); }

// Compara pela DISTANCIA econfere que a distancia reportada bate com o ponto do indice reportado
static bool mesmo_resultado(const std::vector<Vizinho>& a, const std::vector<Vizinho>& b,
                            const Escalar* pts, int n, int dim, const Escalar* q) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!perto(a[i].distancia2, b[i].distancia2)) return false;
        if (a[i].indice < 0 || a[i].indice >= n) return false;
        if (!perto(distancia2(pts + (std::size_t)a[i].indice * dim, q, dim), a[i].distancia2)) return false;
    }
    return true;
}

// Verifica a estrutura da arvore
static bool verifica_no(const ArvoreKD& t, int id, std::vector<char>& visto, std::vector<int>& sub) {
    if (id < 0) return true;
    const std::vector<No>& nos = t.nos();
    if (id >= (int)nos.size()) return false;
    const No& no = nos[id];
    if (no.ponto < 0 || no.ponto >= t.tamanho() || visto[no.ponto]) return false;
    if (no.eixo < 0 || no.eixo >= t.dimensao()) return false;
    visto[no.ponto] = 1;
    std::vector<int> esq, dir;
    if (!verifica_no(t, no.esquerda, visto, esq) || !verifica_no(t, no.direita, visto, dir)) return false;
    const std::size_t d = (std::size_t)t.dimensao();
    Escalar mediana = t.pontos()[(std::size_t)no.ponto * d + no.eixo];
    for (int i : esq) if (t.pontos()[(std::size_t)i * d + no.eixo] > mediana) return false;
    for (int i : dir) if (t.pontos()[(std::size_t)i * d + no.eixo] < mediana) return false;
    sub = esq; sub.insert(sub.end(), dir.begin(), dir.end()); sub.push_back(no.ponto);
    return true;
}

static bool estrutura_valida(const ArvoreKD& t, const Escalar* original) {
    if (t.tamanho() == 0) return true;
    if ((int)t.nos().size() != t.tamanho()) return false;
    if (t.pontos().size() != (std::size_t)t.tamanho() * t.dimensao()) return false;
    for (std::size_t i = 0; i < t.pontos().size(); ++i) if (t.pontos()[i] != original[i]) return false;
    std::vector<char> visto(t.tamanho(), 0);
    std::vector<int> sub;
    if (!verifica_no(t, t.raiz(), visto, sub)) return false;
    for (char v : visto) if (!v) return false;
    return true;
}

static void teste_aleatorio(int dim, int n, int nq) {
    std::vector<Escalar> pts = gerar_uniforme(n, dim, 1);
    std::vector<Escalar> qs  = gerar_uniforme(nq, dim, 2);
    auto t0 = Relogio::now();
    ArvoreKD arv(pts.data(), n, dim);
    auto t1 = Relogio::now();

    VERIFICA(estrutura_valida(arv, pts.data()), "dim=%d: estrutura da arvore invalida", dim);

    int ruins = 0, ruins_nn = 0;
    for (int k : {1, 5, 10})
        for (int i = 0; i < nq; ++i) {
            const Escalar* q = qs.data() + (std::size_t)i * dim;
            if (!mesmo_resultado(arv.k_vizinhos(q, k), k_vizinhos_forca_bruta(pts.data(), n, dim, q, k),
                                 pts.data(), n, dim, q)) ++ruins;
        }
    for (int i = 0; i < nq; ++i) {
        const Escalar* q = qs.data() + (std::size_t)i * dim;
        Vizinho a = arv.vizinho_mais_proximo(q);
        Vizinho b = vizinho_mais_proximo_forca_bruta(pts.data(), n, dim, q);
        if (!perto(a.distancia2, b.distancia2)) ++ruins_nn;
    }
    VERIFICA(ruins == 0, "dim=%d: %d consultas k-NN divergentes", dim, ruins);
    VERIFICA(ruins_nn == 0, "dim=%d: %d consultas 1-NN divergentes", dim, ruins_nn);

    auto a = Relogio::now();
    for (int i = 0; i < nq; ++i) k_vizinhos_forca_bruta(pts.data(), n, dim, qs.data() + (std::size_t)i * dim, 5);
    auto b = Relogio::now();
    for (int i = 0; i < nq; ++i) arv.k_vizinhos(qs.data() + (std::size_t)i * dim, 5);
    auto c = Relogio::now();
    double tb = ms(a, b), tk = ms(b, c);
    std::printf("  d=%-3d n=%d q=%d | build KD %8.2f ms | brute %9.2f ms | kd %9.2f ms | brute/kd = %5.2fx | %s\n",
                dim, n, nq, ms(t0, t1), tb, tk, tb / tk, (ruins || ruins_nn) ? "DIVERGE" : "OK");
}

static void casos_de_borda() {
    std::printf("Casos de borda:\n");
    const int dim = 3;
    { // k > n deve devolver n vizinhos
        auto p = gerar_uniforme(4, dim); ArvoreKD t(p.data(), 4, dim); Escalar q[3] = {.5, .5, .5};
        VERIFICA(t.k_vizinhos(q, 10).size() == 4, "k > n (arvore)");
        VERIFICA(k_vizinhos_forca_bruta(p.data(), 4, dim, q, 10).size() == 4, "k > n (forca bruta)");
    }
    { // n = 1
        auto p = gerar_uniforme(1, dim); ArvoreKD t(p.data(), 1, dim); Escalar q[3] = {0, 0, 0};
        auto r = t.k_vizinhos(q, 1);
        VERIFICA(r.size() == 1 && r[0].indice == 0, "n = 1");
        VERIFICA(t.vizinho_mais_proximo(q).indice == 0, "n = 1 (1-NN)");
    }
    { // pontos todos iguais
        std::vector<Escalar> p(50 * dim, 0.3); ArvoreKD t(p.data(), 50, dim); Escalar q[3] = {.3, .3, .3};
        auto r = t.k_vizinhos(q, 7);
        VERIFICA(r.size() == 7, "duplicados: tamanho");
        for (auto& x : r) VERIFICA(x.distancia2 == 0.0, "duplicados: distancia");
    }
    { // consulta igual a um ponto do conjunto: distancia 0
        auto p = gerar_uniforme(500, dim, 9); ArvoreKD t(p.data(), 500, dim);
        VERIFICA(t.vizinho_mais_proximo(p.data() + 123 * dim).distancia2 == 0.0, "consulta = ponto do conjunto");
    }
    { // 1-NN da arvore e da forca bruta consistentes com k-NN (k=1) da forca bruta
        auto p = gerar_uniforme(300, dim, 5); auto q = gerar_uniforme(1, dim, 6);
        VERIFICA(vizinho_mais_proximo_forca_bruta(p.data(), 300, dim, q.data()).distancia2 ==
                 k_vizinhos_forca_bruta(p.data(), 300, dim, q.data(), 1)[0].distancia2, "1-NN vs k-NN (forca bruta)");
    }
    { // ordenacao: distancias nao decrescentes
        auto p = gerar_uniforme(1000, dim, 11); auto q = gerar_uniforme(1, dim, 12); ArvoreKD t(p.data(), 1000, dim);
        auto r = t.k_vizinhos(q.data(), 20); bool ok = true;
        for (std::size_t i = 1; i < r.size(); ++i) if (r[i].distancia2 < r[i - 1].distancia2) ok = false;
        VERIFICA(ok, "resultado nao esta ordenado do mais perto ao mais longe");
    }
    std::printf("  concluido\n");
}

int main() {
    std::printf("Corretude + baseline (ArvoreKD vs forca bruta)\n");
    for (int dim : {2, 3, 10, 50}) teste_aleatorio(dim, 20000, 200);
    casos_de_borda();
    if (falhas) std::printf("\n%d FALHA(S)\n", falhas);
    else        std::printf("\nTODOS OS TESTES PASSARAM\n");
    return falhas ? 1 : 0;
}
