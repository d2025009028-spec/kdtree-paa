#include "kdtree.h"

#include <algorithm>  // std::nth_element
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

}  // namespace kd
