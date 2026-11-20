#include <iostream>
#include <random>
#include <vector>

#include "kdtree.h"

using namespace kd;

// Imprime a árvore indentada: cada filho fica mais à direita que o pai.
void imprimir(const ArvoreKD& arv, int no, int nivel, const char* lado) {
    if (no == -1) return;
    const No& x = arv.nos()[no];
    for (int i = 0; i < nivel; i++) std::cout << "  ";
    std::cout << lado << "(";
    for (int j = 0; j < arv.dimensao(); j++) {
        if (j > 0) std::cout << ", ";
        std::cout << arv.pontos()[x.ponto * arv.dimensao() + j];
    }
    std::cout << ")  eixo=" << x.eixo << "\n";
    imprimir(arv, x.esquerda, nivel + 1, "E: ");
    imprimir(arv, x.direita, nivel + 1, "D: ");
}

// Todos os pontos da subárvore 'no' respeitam coord[eixo] <= limite
// (se menor_igual) ou >= limite (caso contrário)?
bool confere(const ArvoreKD& arv, int no, int eixo, Escalar limite, bool menor_igual) {
    if (no == -1) return true;
    const No& x = arv.nos()[no];
    Escalar valor = arv.pontos()[x.ponto * arv.dimensao() + eixo];
    bool ok = menor_igual ? (valor <= limite) : (valor >= limite);
    return ok && confere(arv, x.esquerda, eixo, limite, menor_igual)
              && confere(arv, x.direita, eixo, limite, menor_igual);
}

// A propriedade da KD-Tree vale em TODOS os nós?
bool valida(const ArvoreKD& arv, int no) {
    if (no == -1) return true;
    const No& x = arv.nos()[no];
    Escalar mediana = arv.pontos()[x.ponto * arv.dimensao() + x.eixo];
    return confere(arv, x.esquerda, x.eixo, mediana, true)
        && confere(arv, x.direita, x.eixo, mediana, false)
        && valida(arv, x.esquerda) && valida(arv, x.direita);
}

int main() {
    // Teste 1: exemplo pequeno em 2D, para conferir a olho.
    std::vector<Escalar> pequeno = {2, 3,  5, 4,  9, 6,  4, 7,  8, 1,  7, 2};
    ArvoreKD arv(pequeno.data(), 6, 2);
    std::cout << "Arvore do exemplo (6 pontos, 2D):\n";
    imprimir(arv, arv.raiz(), 0, "");
    std::cout << "valida? " << (valida(arv, arv.raiz()) ? "sim" : "NAO") << "\n\n";

    // Teste 2: pontos aleatórios em várias dimensões.
    std::mt19937 gerador(42);
    std::uniform_real_distribution<Escalar> sorteio(0.0, 1.0);
    int dims[] = {2, 3, 10, 50};
    bool tudo_ok = true;
    for (int dim : dims) {
        int n = 1000;
        std::vector<Escalar> pontos(n * dim);
        for (Escalar& v : pontos) v = sorteio(gerador);
        ArvoreKD a(pontos.data(), n, dim);
        bool ok = valida(a, a.raiz()) && static_cast<int>(a.nos().size()) == n;
        std::cout << "d=" << dim << ", n=" << n << ": "
                  << (ok ? "OK" : "FALHOU") << "\n";
        tudo_ok = tudo_ok && ok;
    }
    return tudo_ok ? 0 : 1;
}
