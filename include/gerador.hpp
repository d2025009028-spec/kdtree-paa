#pragma once
#include "kdtree.h"
#include <random>
#include <vector>

namespace kd {


inline std::vector<Escalar> gerar_uniforme(int n, int dim, unsigned long long semente = 42) {
    std::mt19937_64 rng(semente);
    std::uniform_real_distribution<Escalar> U(0.0, 1.0);
    std::vector<Escalar> p((std::size_t)n * dim);
    for (auto& x : p) x = U(rng);
    return p;
}

// normal padrao 
inline std::vector<Escalar> gerar_gaussiano(int n, int dim, unsigned long long semente = 42) {
    std::mt19937_64 rng(semente);
    std::normal_distribution<Escalar> N(0.0, 1.0);
    std::vector<Escalar> p((std::size_t)n * dim);
    for (auto& x : p) x = N(rng);
    return p;
}

}  
