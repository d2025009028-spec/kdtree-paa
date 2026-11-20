#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "kdtree.h"

using namespace kd;

// Busca "na unha": compara a consulta com TODOS os pontos.
// (O Stricker vai fazer a versão oficial; esta é só para conferir.)
Vizinho busca_simples(const std::vector<Escalar>& pontos, int n, int dim,
                      const Escalar* consulta) {
    Vizinho melhor{-1, 1e300};
    for (int i = 0; i < n; i++) {
        Escalar soma = 0;
        for (int j = 0; j < dim; j++) {
            Escalar d = pontos[i * dim + j] - consulta[j];
            soma += d * d;
        }
        if (soma < melhor.distancia2) melhor = Vizinho{i, soma};
    }
    return melhor;
}

int main() {
    // Teste 1: exemplo pequeno em 2D, conferindo a olho.
    std::vector<Escalar> pequeno = {2, 3,  5, 4,  9, 6,  4, 7,  8, 1,  7, 2};
    ArvoreKD arv(pequeno.data(), 6, 2);
    Escalar consulta[2] = {9, 2};
    Vizinho v = arv.vizinho_mais_proximo(consulta);
    std::cout << "Consulta (9, 2): vizinho = ponto " << v.indice
              << ", distancia^2 = " << v.distancia2 << "\n";
    std::cout << "(esperado: ponto 4 = (8, 1) ou ponto 5 = (7, 2); "
                 "distancia^2 = 2 e 4 -> ponto 4)\n\n";

    // Teste 2: compara a KD-Tree com a busca simples em dimensões diferentes.
    std::mt19937 gerador(123);
    std::uniform_real_distribution<Escalar> sorteio(0.0, 1.0);
    int dims[] = {2, 3, 10, 50};
    bool tudo_ok = true;
    for (int dim : dims) {
        int n = 2000, consultas = 300;
        std::vector<Escalar> pontos(n * dim);
        for (Escalar& x : pontos) x = sorteio(gerador);
        ArvoreKD a(pontos.data(), n, dim);

        int erros = 0;
        std::vector<Escalar> q(dim);
        for (int c = 0; c < consultas; c++) {
            for (Escalar& x : q) x = sorteio(gerador);
            Vizinho r1 = a.vizinho_mais_proximo(q.data());
            Vizinho r2 = busca_simples(pontos, n, dim, q.data());
            if (std::fabs(r1.distancia2 - r2.distancia2) > 1e-12) erros++;
        }
        // Consulta igual a um ponto do conjunto: a distância tem de ser 0.
        Vizinho exato = a.vizinho_mais_proximo(&pontos[7 * dim]);
        if (exato.distancia2 != 0) erros++;

        std::cout << "d=" << dim << ": " << consultas << " consultas, "
                  << erros << " erros -> " << (erros == 0 ? "OK" : "FALHOU") << "\n";
        tudo_ok = tudo_ok && (erros == 0);
    }
    return tudo_ok ? 0 : 1;
}
