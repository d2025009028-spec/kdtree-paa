#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "kdtree.h"

using namespace kd;

// Busca "na unha": calcula a distância até TODOS os pontos, ordena e devolve
// os k primeiros. (O Stricker faz a versão oficial; esta é só para conferir.)
std::vector<Vizinho> k_simples(const std::vector<Escalar>& pontos, int n, int dim,
                               const Escalar* consulta, int k) {
    std::vector<Vizinho> todos;
    for (int i = 0; i < n; i++) {
        Escalar soma = 0;
        for (int j = 0; j < dim; j++) {
            Escalar d = pontos[i * dim + j] - consulta[j];
            soma += d * d;
        }
        todos.push_back(Vizinho{i, soma});
    }
    std::sort(todos.begin(), todos.end(),
              [](const Vizinho& a, const Vizinho& b) { return a.distancia2 < b.distancia2; });
    if (static_cast<int>(todos.size()) > k) todos.resize(k);
    return todos;
}

int main() {
    // Teste 1: exemplo pequeno em 2D, conferindo a olho.
    std::vector<Escalar> pequeno = {2, 3,  5, 4,  9, 6,  4, 7,  8, 1,  7, 2};
    ArvoreKD arv(pequeno.data(), 6, 2);
    Escalar consulta[2] = {9, 2};
    std::cout << "Os 3 vizinhos mais proximos de (9, 2):\n";
    for (const Vizinho& v : arv.k_vizinhos(consulta, 3)) {
        std::cout << "  ponto " << v.indice << "  distancia^2 = " << v.distancia2 << "\n";
    }
    std::cout << "(esperado: ponto 4 com 2, ponto 5 com 4, ponto 2 com 16)\n\n";

    // Teste 2: compara a KD-Tree com a busca simples.
    std::mt19937 gerador(7);
    std::uniform_real_distribution<Escalar> sorteio(0.0, 1.0);
    int dims[] = {2, 3, 10, 50};
    int ks[] = {1, 5, 20};
    bool tudo_ok = true;
    for (int dim : dims) {
        int n = 2000, consultas = 200;
        std::vector<Escalar> pontos(n * dim);
        for (Escalar& x : pontos) x = sorteio(gerador);
        ArvoreKD a(pontos.data(), n, dim);

        int erros = 0;
        std::vector<Escalar> q(dim);
        for (int c = 0; c < consultas; c++) {
            for (Escalar& x : q) x = sorteio(gerador);
            for (int k : ks) {
                std::vector<Vizinho> r1 = a.k_vizinhos(q.data(), k);
                std::vector<Vizinho> r2 = k_simples(pontos, n, dim, q.data(), k);
                if (r1.size() != r2.size()) { erros++; continue; }
                for (size_t i = 0; i < r1.size(); i++) {
                    if (std::fabs(r1[i].distancia2 - r2[i].distancia2) > 1e-12) { erros++; break; }
                }
            }
            // k = 1 tem de dar o mesmo resultado do vizinho_mais_proximo.
            if (a.k_vizinhos(q.data(), 1)[0].indice != a.vizinho_mais_proximo(q.data()).indice) erros++;
        }
        // k maior que n: devolve todos os n pontos, ordenados.
        std::vector<Vizinho> todos = a.k_vizinhos(q.data(), n + 10);
        if (static_cast<int>(todos.size()) != n) erros++;
        for (size_t i = 1; i < todos.size(); i++)
            if (todos[i - 1].distancia2 > todos[i].distancia2) { erros++; break; }

        std::cout << "d=" << dim << ": " << erros << " erros -> "
                  << (erros == 0 ? "OK" : "FALHOU") << "\n";
        tudo_ok = tudo_ok && (erros == 0);
    }
    return tudo_ok ? 0 : 1;
}
