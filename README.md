CTC04 — Tema 1.3: KD-Tree (Sequencial, Força Bruta, OpenMP e CUDA)
Implementação e avaliação experimental de busca exata dos vizinhos mais próximos (1-NN e k-NN) utilizando KD-Tree e Força Bruta em versões sequencial (CPU), paralela em memória compartilhada (OpenMP) e paralela em acelerador gráfico (CUDA).
Divisão do Trabalho
Cauã: construção da KD-Tree (seleção do eixo de corte e mediana) e busca sequencial 1-NN e k-NN.
Stricker: implementação linear de força bruta (baseline), gerador de pontos aleatórios (d \in \{2, 3, 10, 50\}) e testes de corretude comparando KD-Tree vs. força bruta.
Davi: paralelização em CPU (OpenMP) e GPU (CUDA) e execução dos benchmarks de speedup.
Eduardo: análise da maldição da dimensionalidade, scripts/gráficos/tabelas de análise, README e referências.
Estrutura do Projeto
include/: arquivos de cabeçalho (.h / .hpp / .cuh) com as estruturas de dados e interfaces da KD-Tree e dos algoritmos de busca.
src/: implementação principal da construção e busca da KD-Tree (kdtree.cpp).
tests/: suíte de testes unitários e de corretude (versões sequencial, OpenMP e CUDA).
benchmarks/: códigos de medição de desempenho, escalabilidade por threads e comparação final entre abordagens.
resultados/: arquivos .csv com os dados brutos e consolidados gerados pelos benchmarks.
Requisitos de Compilação e Execução
Ambiente: Windows (os comandos que utilizam nvcc devem ser executados no x64 Native Tools Command Prompt).
CPU / OpenMP: compilador g++ com suporte a C++17 e OpenMP (-fopenmp).
GPU / CUDA: NVIDIA CUDA Toolkit, Visual Studio Build Tools e GPU NVIDIA compatível.
Nota sobre arquitetura: os comandos abaixo utilizam -arch=sm_89 (arquitetura Ada Lovelace, referente à NVIDIA GeForce RTX 4050 usada nos testes). Caso compile em outra GPU, ajuste a flag -arch=sm_XX de acordo com a Compute Capability do seu dispositivo.
Comandos de Compilação e Execução (Windows)
1. Testes Sequenciais (Construção, 1-NN, k-NN e Corretude)
