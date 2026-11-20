#pragma once
#include <vector>

namespace kd {

using Escalar = double;

//qual é ponto e distância
struct Vizinho {
    int     indice;      // índice original
    Escalar distancia2;  // distância ao quadrado euclidiana 
};

// nó da árvore, os filhos são índices no vetor dos nós (eu usei o -1 pra dizer q n existe)
struct No {
    int ponto;     // índice do ponto que está nesse nó
    int eixo;      // eixo de corte
    int esquerda;  // filho esquerdo (valores menores do que a mediana no eixo)
    int direita;   // filho direito  (valores maiores do que a mediana no eixo)
};

class ArvoreKD {
public:
    // pontos: vetor dos n*dim valores seguidos, o ponto i ocupa [i*dim, (i+1)*dim)
    // essa árvore guarda uma cópia dos pontos
    ArvoreKD(const Escalar* pontos, int n, int dim);

    // vizinho mais próximo 1-NN
    Vizinho vizinho_mais_proximo(const Escalar* consulta) const;

    // k vizinhos mais próximo do mais perto até o mais longe
    std::vector<Vizinho> k_vizinhos(const Escalar* consulta, int k) const;

    int tamanho()  const { return n_; }
    int dimensao() const { return dim_; }
    int raiz()     const { return raiz_; }

    // acessa a estrutura para os testes
    const std::vector<No>&      nos()    const { return nos_; }
    const std::vector<Escalar>& pontos() const { return pontos_; }

private:
    int n_, dim_, raiz_;
    std::vector<Escalar> pontos_;
    std::vector<No>      nos_;
    // auxiliar
};

}