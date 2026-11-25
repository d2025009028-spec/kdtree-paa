# CTC04 — Tema 1.3: KD-Tree (Sequencial, Força Bruta, OpenMP e CUDA)

Implementação e avaliação experimental de busca exata dos vizinhos mais próximos ($1$-NN e $k$-NN) utilizando **KD-Tree** e **Força Bruta** em versões sequencial (CPU), paralela em memória compartilhada (**OpenMP**) e paralela em acelerador gráfico (**CUDA**).

## Divisão do Trabalho
- **Cauã:** construção da KD-Tree (seleção do eixo de corte e mediana) e busca sequencial $1$-NN e $k$-NN.
- **Stricker:** implementação linear de força bruta (*baseline*), gerador de pontos aleatórios ($d \in \{2, 3, 10, 50\}$) e testes de corretude comparando KD-Tree vs. força bruta.
- **Davi:** paralelização em CPU (OpenMP) e GPU (CUDA) e execução dos benchmarks de *speedup*.
- **Eduardo:** análise da maldição da dimensionalidade, scripts/gráficos/tabelas de análise, README e referências.

---

## Estrutura do Projeto
- `include/`: arquivos de cabeçalho (`.h` / `.hpp` / `.cuh`) com as estruturas de dados e interfaces da KD-Tree e dos algoritmos de busca.
- `src/`: implementação principal da construção e busca da KD-Tree (`kdtree.cpp`).
- `tests/`: suíte de testes unitários e de corretude (versões sequencial, OpenMP e CUDA).
- `benchmarks/`: códigos de medição de desempenho, escalabilidade por threads e comparação final entre abordagens.
- `resultados/`: arquivos `.csv` com os dados brutos e consolidados gerados pelos benchmarks.

---

## Requisitos de Compilação e Execução
- **Ambiente:** Windows (os comandos que utilizam `nvcc` devem ser executados no **x64 Native Tools Command Prompt**).
- **CPU / OpenMP:** compilador `g++` com suporte a C++17 e OpenMP (`-fopenmp`).
- **GPU / CUDA:** NVIDIA CUDA Toolkit, Visual Studio Build Tools e GPU NVIDIA compatível.
  - *Nota sobre arquitetura:* os comandos abaixo utilizam `-arch=sm_89` (arquitetura Ada Lovelace, referente à **NVIDIA GeForce RTX 4050** usada nos testes). Caso compile em outra GPU, ajuste a *flag* `-arch=sm_XX` de acordo com a *Compute Capability* do seu dispositivo.

---

## Comandos de Compilação e Execução (Windows)

### 1. Testes Sequenciais (Construção, 1-NN, k-NN e Corretude)
```bat
g++ -O2 -std=c++17 -Iinclude src/kdtree.cpp tests/teste_construcao.cpp -o teste_construcao.exe
teste_construcao.exe

g++ -O2 -std=c++17 -Iinclude src/kdtree.cpp tests/teste_1nn.cpp -o teste_1nn.exe
teste_1nn.exe

g++ -O2 -std=c++17 -Iinclude src/kdtree.cpp tests/teste_knn.cpp -o teste_knn.exe
teste_knn.exe

g++ -O2 -std=c++17 -Iinclude src/kdtree.cpp tests/teste_corretude.cpp -o teste_corretude.exe
teste_corretude.exe
```

### 2. Teste Paralelo OpenMP
```bat
g++ -O2 -std=c++17 -fopenmp -Iinclude src/kdtree.cpp tests/teste_paralelo.cpp -o teste_paralelo.exe
teste_paralelo.exe
```

### 3. Teste GPU (CUDA)
```bat
nvcc -O3 -arch=sm_89 -std=c++17 -Iinclude src/kdtree.cpp tests/teste_gpu.cu -o teste_gpu.exe
teste_gpu.exe
```

### 4. Benchmark de Escalabilidade por Threads (OpenMP)
```bat
g++ -O2 -std=c++17 -fopenmp -Iinclude src/kdtree.cpp benchmarks/bench_threads.cpp -o bench_threads.exe
bench_threads.exe
```

### 5. Benchmark Final (Sequencial, OpenMP e CUDA)
```bat
nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/O2,/openmp" -Iinclude src/kdtree.cpp benchmarks/bench_final.cu -o bench_final.exe
bench_final.exe 3 1.0 7 saida.csv
```

> **Atenção:** Sem as flags `-O3` e `/O2`, o `nvcc` não otimiza adequadamente o código executado na CPU (*host*), fazendo com que os tempos fiquem entre **1,2× e 1,8× piores**.

---

## Máquina e Configuração de Testes
- **Processador:** Intel Core i7-12650HX (20 threads)
- **Memória RAM:** 16 GB
- **GPU:** NVIDIA GeForce RTX 4050 Laptop (6 GB VRAM, `sm_89`)
- **Sistema Operacional e Energia:** Windows 11, notebook conectado na tomada em modo *Melhor desempenho*

---

## Estrutura dos Resultados (`resultados/`)
- `final_run1.csv`, `final_run2.csv`, `final_run3.csv`: saídas brutas das 3 rodadas executadas com `bench_final.exe`.
- `resultados_final_mediana.csv`: consolidação estatística com mediana, mínimo, máximo e `speedup_vs_seq` para cada método e dimensionalidade (utilizado nas tabelas e gráficos do relatório).
- `resultados_threads.csv`: tempos e fatores de *speedup* da versão OpenMP variando de 1 a 20 threads.

---

## Referências
1. Bentley, J. L. (1975). *Multidimensional binary search trees used for associative searching*. Communications of the ACM, 18(9), 509–517.
2. Friedman, J. H., Bentley, J. L., & Finkel, R. A. (1977). *An algorithm for finding best matches in logarithmic expected time*. ACM Transactions on Mathematical Software (TOMS), 3(3), 209–226.
3. Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). *When is “nearest neighbor” meaningful?* In Database Theory — ICDT’99 (pp. 217–235). Springer.
4. NVIDIA Corporation (2024). *CUDA C++ Programming Guide*. Disponível em: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
5. OpenMP Architecture Review Board (2021). *OpenMP Application Program Interface (Version 5.2)*. Disponível em: https://www.openmp.org/specifications/
