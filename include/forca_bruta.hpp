#pragma once
#include "kdtree.h"
#include <queue>
#include <vector>

namespace kd {

// distancia euclidiana ao quadrado entre dois pontos 
inline Escalar distancia2(const Escalar* a, const Escalar* b, int dim) {
    Escalar s = 0;
    for (int i = 0; i < dim; ++i) { Escalar t = a[i] - b[i]; s += t * t; }
    return s;
}

// ordem canonica: menor distancia primeiro; desempate no menor indice
inline bool vizinho_menor(const Vizinho& a, const Vizinho& b) {
    return a.distancia2 < b.distancia2 ||
           (a.distancia2 == b.distancia2 && a.indice < b.indice);
}

struct ComparaVizinho {
    bool operator()(const Vizinho& a, const Vizinho& b) const { return vizinho_menor(a, b); }
};

//Retorna ate k vizinhos, do mais perto ao mais longe.
inline std::vector<Vizinho> k_vizinhos_forca_bruta(const Escalar* pontos, int n, int dim,
                                                   const Escalar* consulta, int k) {
    std::priority_queue<Vizinho, std::vector<Vizinho>, ComparaVizinho> heap;
    for (int i = 0; i < n; ++i) {
        Vizinho c{i, distancia2(pontos + (std::size_t)i * dim, consulta, dim)};
        if ((int)heap.size() < k) heap.push(c);
        else if (k > 0 && vizinho_menor(c, heap.top())) { heap.pop(); heap.push(c); }
    }
    std::vector<Vizinho> out(heap.size());
    for (std::size_t i = out.size(); i-- > 0;) { out[i] = heap.top(); heap.pop(); }
    return out;
}


inline Vizinho vizinho_mais_proximo_forca_bruta(const Escalar* pontos, int n, int dim,
                                                const Escalar* consulta) {
    Vizinho melhor{0, distancia2(pontos, consulta, dim)};
    for (int i = 1; i < n; ++i) {
        Vizinho c{i, distancia2(pontos + (std::size_t)i * dim, consulta, dim)};
        if (vizinho_menor(c, melhor)) melhor = c;
    }
    return melhor;
}

}  
