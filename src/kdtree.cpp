#include "kdtree.h"

#include <algorithm>  // std::nth_element
#include <limits>     // std::numeric_limits (infinito)
#include <numeric>    // std::iota

namespace kd {

namespace {  // funções "privadas" deste arquivo (ninguém de fora enxerga)

// Constrói a subárvore formada pelos pontos indices[inicio ... fim-1].
// Devolve a posição do nó-raiz dessa subárvore dentro de 'nos',
// ou -1 se não há pontos para colocar.
int construir(const std::vector<Escalar>& pontos, int dim,
              std::vector<int>& indices, int inicio, int fim,
              int profundidade, std::vector<No>& nos) {

    // Caso base: faixa vazia -> não existe subárvore.
    if (inicio >= fim) return -1;

    // 1) Escolha do eixo de corte: alterna x, y, z, x, y, z...
    int eixo = profundidade % dim;

    // 2) Mediana: reorganiza a faixa para que o elemento do meio fique
    //    na posição certa, com os menores antes dele e os maiores depois.
    int meio = (inicio + fim) / 2;
    std::nth_element(indices.begin() + inicio,
                     indices.begin() + meio,
                     indices.begin() + fim,
                     [&](int a, int b) {
                         return pontos[a * dim + eixo] < pontos[b * dim + eixo];
                     });

    // 3) Cria o nó com o ponto da mediana. Guardamos a POSIÇÃO (e não uma
    //    referência), porque o vetor pode ser realocado nas chamadas abaixo.
    int posicao = static_cast<int>(nos.size());
    nos.push_back(No{indices[meio], eixo, -1, -1});

    // 4) Recursão: esquerda recebe os menores, direita os maiores.
    int esq = construir(pontos, dim, indices, inicio, meio, profundidade + 1, nos);
    int dir = construir(pontos, dim, indices, meio + 1, fim, profundidade + 1, nos);
    nos[posicao].esquerda = esq;
    nos[posicao].direita = dir;

    return posicao;
}

// Distância euclidiana AO QUADRADO entre dois pontos de 'dim' coordenadas.
// (Sem raiz quadrada: comparar quadrados dá o mesmo resultado e é mais rápido.)
Escalar distancia2(const Escalar* a, const Escalar* b, int dim) {
    Escalar soma = 0;
    for (int j = 0; j < dim; j++) {
        Escalar d = a[j] - b[j];
        soma += d * d;
    }
    return soma;
}

// Busca recursiva do vizinho mais próximo na subárvore que começa em 'no'.
// 'melhor' guarda o melhor candidato encontrado até agora e é atualizado.
void buscar_1nn(const std::vector<Escalar>& pontos, int dim,
                const std::vector<No>& nos, int no,
                const Escalar* consulta, Vizinho& melhor) {

    // Caso base: chegou no fim de um galho.
    if (no == -1) return;

    const No& x = nos[no];
    const Escalar* p = &pontos[x.ponto * dim];  // coordenadas deste nó

    // 1) O ponto deste nó é melhor que o melhor até agora?
    Escalar d2 = distancia2(p, consulta, dim);
    if (d2 < melhor.distancia2) {
        melhor.indice = x.ponto;
        melhor.distancia2 = d2;
    }

    // 2) De que lado do corte a consulta cai? Esse é o lado "perto".
    Escalar diferenca = consulta[x.eixo] - p[x.eixo];
    int perto = (diferenca < 0) ? x.esquerda : x.direita;
    int longe = (diferenca < 0) ? x.direita : x.esquerda;

    // 3) Sempre explora primeiro o lado perto.
    buscar_1nn(pontos, dim, nos, perto, consulta, melhor);

    // 4) PODA: o lado longe só pode ter algo melhor se o plano de corte
    //    estiver mais perto da consulta do que o melhor já encontrado.
    if (diferenca * diferenca < melhor.distancia2) {
        buscar_1nn(pontos, dim, nos, longe, consulta, melhor);
    }
}

}  // namespace

ArvoreKD::ArvoreKD(const Escalar* pontos, int n, int dim)
    : n_(n), dim_(dim), raiz_(-1),
      pontos_(pontos, pontos + n * dim) {  // copia os n*dim números

    // Lista 0, 1, 2, ..., n-1: vamos reorganizar ÍNDICES, não os pontos.
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    nos_.reserve(n);
    raiz_ = construir(pontos_, dim_, indices, 0, n_, 0, nos_);
}

Vizinho ArvoreKD::vizinho_mais_proximo(const Escalar* consulta) const {
    // Começa com um "melhor" infinitamente longe: qualquer ponto o supera.
    Vizinho melhor{-1, std::numeric_limits<Escalar>::infinity()};
    buscar_1nn(pontos_, dim_, nos_, raiz_, consulta, melhor);
    return melhor;
}

}  // namespace kd
