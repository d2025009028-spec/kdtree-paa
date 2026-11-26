# KD-Tree: vizinho mais próximo (1-NN e k-NN)

Trabalho prático de PAA (CTC04, Prof. Sávio S. Dias), tema 1.3.
Equipe: Cauã, Davi, Eduardo e Stricker.

A gente implementou uma KD-Tree em C++ para achar o vizinho mais próximo de um ponto (1-NN) e os k vizinhos mais próximos (k-NN). Depois comparamos com a força bruta, paralelizamos com OpenMP e CUDA e medimos o que acontece quando a dimensão aumenta (a "maldição da dimensionalidade").

Resumo do que descobrimos: a KD-Tree é muito boa em dimensão baixa (d = 2 e 3, de 110× a 134× mais rápida que a força bruta) e perde para ela em d = 50. O relatório explica o porquê, com números.

## O que precisa ter instalado

- Um compilador C++17 (`g++`). Para os testes com OpenMP, o `g++` precisa aceitar `-fopenmp`.
- Para a parte de GPU: CUDA Toolkit (`nvcc`) e uma placa NVIDIA. No Windows, o `nvcc` também precisa do Visual Studio (compilador `cl`).
- Para os gráficos e tabelas: Python 3 com `numpy`, `pandas` e `matplotlib`.

## Como o projeto está organizado

| Pasta | O que tem |
|---|---|
| `include/` | `kdtree.h` (a interface da árvore), `forca_bruta.hpp` (baseline) e `gerador.hpp` (gerador de pontos) |
| `src/` | `kdtree.cpp`: construção da árvore e buscas 1-NN e k-NN |
| `paralelo/` | versões com OpenMP (`omp_lote.h`, `omp_knn.cpp`) |
| `gpu/` | kernels CUDA (`gpu_kernels.cuh`) |
| `tests/` | testes de corretude |
| `benchmarks/` | programas que medem tempo |
| `scripts/` | scripts Python que geram gráficos e tabelas |
| `resultados/` | CSVs com as medições |
| `tabelas/` e `graficos/` | tabelas e gráficos prontos para o relatório |

## Como compilar e rodar os testes

Rode tudo na raiz do projeto. Os testes sequenciais têm cada um o seu `main`, então compile um de cada vez:

```bash
g++ -std=c++17 -O2 -Wall -Iinclude src/kdtree.cpp tests/teste_construcao.cpp -o teste_construcao
g++ -std=c++17 -O2 -Wall -Iinclude src/kdtree.cpp tests/teste_1nn.cpp -o teste_1nn
g++ -std=c++17 -O2 -Wall -Iinclude src/kdtree.cpp tests/teste_knn.cpp -o teste_knn
g++ -std=c++17 -O2 -Wall -Iinclude src/kdtree.cpp tests/teste_corretude.cpp -o teste_corretude
```

Teste do OpenMP (confere que o resultado paralelo é igual ao sequencial com 1, 2, 3, 4, 8 e 16 threads):

```bash
g++ -std=c++17 -O2 -fopenmp -Iinclude src/kdtree.cpp tests/teste_paralelo.cpp -o teste_paralelo
```

Teste da GPU (precisa de CUDA):

```bash
nvcc -O3 -arch=sm_89 -std=c++17 -Iinclude src/kdtree.cpp tests/teste_gpu.cu -o teste_gpu
```

Todos imprimem os testes que passaram no final. Se algum falhar, ele diz qual.

> **Atenção:** `-arch=sm_89` é para a RTX 4050 do notebook onde a gente mediu. Se a sua placa for outra, troque pelo valor dela (por exemplo `sm_86` para uma RTX 3060).

## Como rodar os benchmarks

**Comparação KD-Tree × força bruta, com contagem de nós visitados** (sequencial):

```bash
g++ -O2 -std=c++17 -Iinclude src/kdtree.cpp benchmarks/baseline_e_poda.cpp -o baseline_e_poda
./baseline_e_poda 1.0 3 resultados/baseline_e_poda.csv
```

Os argumentos são a escala (1.0 = tamanho completo), o número de repetições e o arquivo de saída.

**OpenMP por número de threads** (de 1 a 20):

```bash
g++ -O2 -std=c++17 -fopenmp -Iinclude src/kdtree.cpp benchmarks/bench_threads.cpp -o bench_threads
./bench_threads 3 1.0
```

Os argumentos são as repetições e a escala.

**Benchmark principal** (CPU sequencial, OpenMP, GPU com KD-Tree e GPU com força bruta, em `double` e `float`):

```bash
nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/O2,/openmp" -Iinclude src/kdtree.cpp benchmarks/bench_final.cu -o bench_final
./bench_final 3 1.0 7 resultados/saida.csv
```

Os argumentos são: repetições na CPU, escala, repetições na GPU e arquivo de saída. As opções `-Xcompiler "/O2,/openmp"` são do Windows. Sem elas, o código da CPU compila sem otimização e os tempos pioram de 1,2× a 1,8×. No Linux, o equivalente é `-Xcompiler "-O2,-fopenmp"`.

**Gráficos e tabelas** (leem os CSVs de `resultados/`):

```bash
python scripts/gerar_graficos.py
python scripts/experimento_dimensionalidade.py
```

## Como foi feita a medição

- Máquina: notebook com Intel Core i7-12650HX (20 threads), 16 GB de RAM e NVIDIA RTX 4050 Laptop (6 GB), Windows 11, ligado na tomada em modo Melhor desempenho.
- Dados: 100.000 pontos uniformes em [0,1]^d, semente 42, d = 2, 3, 10 e 50.
- Consultas: 200.000 (d = 2 e 3), 20.000 (d = 10) e 1.000 (d = 50).
- Cada medição é a mediana de várias repetições (3 na CPU, 7 na GPU), e o resultado final é a mediana de 3 execuções do programa (`resultados/final_run1.csv` a `final_run3.csv`).
- O tempo da GPU inclui copiar dados para a placa e trazer a resposta de volta. A construção da árvore não é cronometrada.
- A comparação KD-Tree × força bruta (`baseline_e_poda.csv`) usa só uma amostra de consultas (5.000 em d = 2 e 3, 2.000 em d = 10 e 200 em d = 50) e roda com 1 thread. Os tempos de d = 2 e 3 saem maiores que os do benchmark principal, então essas duas razões têm alguma incerteza. Uma rodada anterior, feita numa máquina virtual Linux, está em `baseline_e_poda_nuvem.csv`.

## Cuidados ao ler os resultados

- Em d = 50 foram só 1.000 consultas, o que é pouco para ocupar bem a GPU. Os speedups dessa dimensão são um piso.
- Só medimos 1-NN. O k-NN está implementado e testado, mas não entrou nos benchmarks de GPU.
- Só usamos dados uniformes, e n ficou fixo em 100.000.

A lista completa de limitações está na seção 8 do relatório.

## Referências

1. Bellman, R. *Adaptive Control Processes: A Guided Tour*. Princeton University Press, 1961.
2. Bentley, J. L. Multidimensional binary search trees used for associative searching. *Communications of the ACM*, 18(9), p. 509–517, 1975.
3. Beyer, K.; Goldstein, J.; Ramakrishnan, R.; Shaft, U. When is "nearest neighbor" meaningful? In: *ICDT'99*, Springer, p. 217–235, 1999.
4. Friedman, J. H.; Bentley, J. L.; Finkel, R. A. An algorithm for finding best matches in logarithmic expected time. *ACM Transactions on Mathematical Software*, 3(3), p. 209–226, 1977.
5. NVIDIA. *CUDA C++ Programming Guide*.
6. OpenMP Architecture Review Board. *OpenMP Application Programming Interface*.
7. Weber, R.; Schek, H.-J.; Blott, S. A quantitative analysis and performance study for similarity-search methods in high-dimensional spaces. In: *VLDB*, p. 194–205, 1998.
