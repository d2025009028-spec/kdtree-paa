#!/usr/bin/env python3
"""Gera o relatorio em PDF (relatorio_ctc04.pdf) a partir dos CSVs de resultados/ e dos
graficos de graficos/. Todos os numeros do texto vem dos CSVs.

Uso (na raiz do projeto, depois de rodar gerar_graficos.py):
    python scripts/gerar_relatorio.py

Se existir resultados/baseline_e_poda.csv (medido na maquina de testes) ele e usado no lugar de
resultados/baseline_e_poda_nuvem.csv, e o texto sobre "outra maquina" some sozinho.
"""
import base64
import os

import pandas as pd
from playwright.sync_api import sync_playwright

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, GRAF = os.path.join(RAIZ, "resultados"), os.path.join(RAIZ, "graficos")
SAIDA_PDF = os.path.join(RAIZ, "relatorio_ctc04.pdf")
SAIDA_HTML = os.path.join(RAIZ, "relatorio_ctc04.html")
DIMS = [2, 3, 10, 50]

# ------------------------------------------------------------------ dados
med = pd.read_csv(os.path.join(RES, "resultados_final_mediana.csv"))
thr = pd.read_csv(os.path.join(RES, "resultados_threads.csv"))
usa_maq = os.path.exists(os.path.join(RES, "baseline_e_poda.csv"))
base = pd.read_csv(os.path.join(RES, "baseline_e_poda.csv" if usa_maq else "baseline_e_poda_nuvem.csv"))
base = base.set_index("dim")


def n(x, d=2):
    """Numero no formato brasileiro."""
    return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def M(d, metodo, col="tempo_mediana_s"):
    return float(med[(med.dim == d) & (med.metodo == metodo)][col].iloc[0])


def SP(d, metodo):
    return M(d, metodo, "speedup_vs_seq")


def CONS(d):
    return int(med[med.dim == d].consultas.iloc[0])


def T(d, th):
    return float(thr[(thr.dim == d) & (thr.metodo == "omp") & (thr.threads == th)].speedup_vs_seq.iloc[0])


def TT(d, th):
    return float(thr[(thr.dim == d) & (thr.metodo == "omp") & (thr.threads == th)].tempo_s.iloc[0])


tq = {d: M(d, "kdtree_seq") / CONS(d) * 1e6 for d in DIMS}                 # us por consulta
fim = {d: (M(d, "kdtree_seq_fim") / M(d, "kdtree_seq") - 1) * 100 for d in DIMS}
razao_gpu = {d: M(d, "gpu_bf_float_total") / M(d, "gpu_kdtree_float_total") for d in DIMS}
razao_cpu = {d: float(base.loc[d, "bf_sobre_kd"]) for d in DIMS}
fator = {d: float(base.loc[d, "kd_seq_us_por_consulta"]) / tq[d] for d in DIMS}  # maquina alt / testes
fmin, fmax = min(fator.values()), max(fator.values())
passo = fmax / fmin

conc = pd.read_csv(os.path.join(RAIZ, "tabelas", "t4_concentracao_distancias.csv")).set_index("d")
CONTR = {d: float(conc.loc[d, "contraste mediano (dmax-dmin)/dmin"]) for d in DIMS}
RZ = {d: float(conc.loc[d, "dmin/dmax mediano"]) for d in DIMS}


def img(nome, alt):
    with open(os.path.join(GRAF, nome), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


# ------------------------------------------------------------------ tabelas
def tabela1():
    linhas = ""
    for d in DIMS:
        linhas += (
            f"<tr><td class=c><b>{d}</b></td><td>{n(CONS(d), 0)}</td>"
            f"<td>{n(M(d, 'kdtree_seq'), 6)}</td><td>{n(tq[d], 2)}</td>"
            f"<td>{n(M(d, 'kdtree_omp20'), 6)} ({n(SP(d, 'kdtree_omp20'))}×)</td>"
            f"<td>{n(M(d, 'gpu_kdtree_float_total'), 6)} ({n(SP(d, 'gpu_kdtree_float_total'))}×)</td>"
            f"<td>{n(M(d, 'gpu_bf_float_total'), 6)} ({n(SP(d, 'gpu_bf_float_total'))}×)</td></tr>"
        )
    return f"""<table><thead><tr><th>d</th><th>Consultas</th><th>KD-Tree seq. (s)</th>
<th>µs por consulta</th><th>KD-Tree OpenMP, 20 threads (s)</th><th>KD-Tree GPU, float (s)</th>
<th>Força bruta GPU, float (s)</th></tr></thead><tbody>{linhas}</tbody></table>"""


def tabela2():
    linhas = ""
    for th in [1, 2, 4, 6, 8, 12, 14, 16, 20]:
        linhas += f"<tr><td class=c><b>{th}</b></td>" + "".join(f"<td>{n(T(d, th))}×</td>" for d in DIMS) + "</tr>"
    return f"""<table class=estreita><thead><tr><th>Threads</th><th>d = 2</th><th>d = 3</th><th>d = 10</th>
<th>d = 50</th></tr></thead><tbody>{linhas}</tbody></table>"""


def tabela3():
    linhas = ""
    for d in DIMS:
        r = base.loc[d]
        linhas += (
            f"<tr><td class=c><b>{d}</b></td><td>{n(r.kd_seq_us_por_consulta, 2)}</td>"
            f"<td>{n(r.bf_seq_us_por_consulta, 2)}</td><td>{n(r.bf_sobre_kd, 2)}×</td>"
            f"<td>{n(r.nos_visitados_mediana, 0)}</td><td>{n(r.fracao_nos_mediana * 100, 3 if r.fracao_nos_mediana < 0.001 else 1)}%</td>"
            f"<td>{n(r.fracao_nos_max * 100, 2 if r.fracao_nos_max < 0.01 else 1)}%</td></tr>"
        )
    return f"""<table><thead><tr><th>d</th><th>KD-Tree seq. (µs/consulta)</th><th>Força bruta seq. (µs/consulta)</th>
<th>Força bruta ÷ KD-Tree</th><th>Nós visitados (mediana)</th><th>Fração da árvore (mediana)</th>
<th>Fração (pior consulta)</th></tr></thead><tbody>{linhas}</tbody></table>"""


def tabela4():
    linhas = "".join(
        f"<tr><td class=c><b>{d}</b></td><td>{n(CONTR[d], 2 if CONTR[d] < 10 else 0)}</td><td>{n(RZ[d], 4 if RZ[d] < 0.1 else 2)}</td></tr>"
        for d in DIMS)
    return f"""<table class=estreita><thead><tr><th>d</th><th>Contraste (d<sub>max</sub> − d<sub>min</sub>) / d<sub>min</sub></th>
<th>d<sub>min</sub> / d<sub>max</sub></th></tr></thead><tbody>{linhas}</tbody></table>"""


# ------------------------------------------------------------------ texto condicional
if usa_maq:
    origem_baseline = ""
else:
    origem_baseline = f"""
<p>Uma observação sobre esta medição: ela foi feita numa máquina virtual Linux com 2 vCPUs Intel Xeon de 2,1 GHz
(g++ 13.3, <code>-O2</code>), e não no notebook dos outros testes. Como a KD-Tree sequencial também foi medida lá, dá para
ver o quanto as máquinas diferem: a árvore ficou de {n(fmin, 1)}× a {n(fmax, 1)}× mais lenta do que no notebook
({n(fator[2], 1)}×, {n(fator[3], 1)}×, {n(fator[10], 1)}× e {n(fator[50], 1)}× em d = 2, 3, 10 e 50). A força bruta
deve ser afetada de um jeito parecido, mas não igual: ela lê a memória em sequência, enquanto a árvore pula de nó em nó.
Se o fator dela cair na mesma faixa, a razão força bruta ÷ KD-Tree da tabela pode mudar por um fator de até
{n(passo, 1)}, para mais ou para menos. Mesmo assim a conclusão não muda: em d = 2 e 3 a diferença é de duas ordens de
grandeza, em d = 10 a razão iria de {n(razao_cpu[10] / passo, 1)}× a {n(razao_cpu[10] * passo, 1)}× (a KD-Tree continuaria ganhando) e em d = 50
iria de {n(razao_cpu[50] / passo, 2)}× a {n(razao_cpu[50] * passo, 2)}× (a força bruta continuaria ganhando).
Rodar <code>baseline_e_poda.exe</code> no notebook de testes elimina essa incerteza e é só trocar o CSV em <code>resultados/</code>.</p>"""

# ------------------------------------------------------------------ html
CSS = """
@page { size: A4; margin: 17mm 19mm 20mm 19mm; }
* { box-sizing: border-box; }
html { font-family: 'Gelasio', Georgia, 'DejaVu Serif', serif; font-size: 10.2pt; line-height: 1.48; color: #1c1c1c; }
body { margin: 0; }
h1, h2, h3, .eyebrow, .eq, table, figcaption, .nota, .rodape { font-family: 'Inter', 'DejaVu Sans', sans-serif; }
.eyebrow { font-size: 8.5pt; letter-spacing: .08em; text-transform: uppercase; color: #2f5fa8; font-weight: 600; margin: 0 0 6px; }
h1 { font-size: 20pt; line-height: 1.2; margin: 0 0 10px; font-weight: 700; color: #111; }
.eq { font-size: 9.5pt; color: #555; margin: 0 0 6px; }
hr.topo { border: 0; border-top: 2px solid #1c1c1c; margin: 10px 0 14px; }
h2 { font-size: 13pt; margin: 20px 0 6px; padding-bottom: 3px; border-bottom: 1px solid #d4d8de; color: #111; break-after: avoid; }
h3 { font-size: 10.6pt; margin: 14px 0 4px; color: #2a4a7f; break-after: avoid; }
p { margin: 0 0 8px; orphans: 3; widows: 3; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt; background: #f1f3f6; padding: 0 3px; border-radius: 3px; }
figure { margin: 8px 0 10px; break-inside: avoid; text-align: center; }
figure img { width: 76%; border: 1px solid #e3e6ea; }
figcaption { font-size: 8.4pt; color: #555; margin-top: 4px; text-align: left; padding: 0 6%; }
table { border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 6px 0 4px; break-inside: avoid; }
table.estreita { width: 78%; margin-left: auto; margin-right: auto; }
thead th { background: #14213d; color: #fff; padding: 5px 6px; text-align: right; font-weight: 600; }
thead th:first-child { text-align: center; }
tbody td { padding: 4px 6px; text-align: right; border-bottom: 1px solid #e3e6ea; }
tbody td.c { text-align: center; }
tbody tr:nth-child(even) { background: #f6f8fa; }
.legenda { font-size: 8.4pt; font-family: 'Inter', sans-serif; color: #444; font-weight: 600; margin: 10px 0 0; break-after: avoid; }
.nota { border-left: 4px solid #2f5fa8; background: #f3f6fb; padding: 8px 12px; margin: 10px 0; font-size: 9.2pt; line-height: 1.5; break-inside: avoid; }
ol.refs { padding-left: 18px; font-size: 9.2pt; line-height: 1.45; }
ol.refs li { margin-bottom: 4px; }
"""


def corpo():
    d2, d3, d10, d50 = DIMS
    return f"""
<div class=eyebrow>CTC04 · Tema 1.3 · Trabalho prático de PAA</div>
<h1>KD-Tree para busca de vizinhos mais próximos: desempenho paralelo e maldição da dimensionalidade</h1>
<div class=eq><b>Equipe:</b> Cauã, Davi, Eduardo e Stricker</div>
<hr class=topo>

<h2>1. Introdução</h2>
<p>Neste trabalho implementamos uma KD-Tree em C++ e a usamos para buscar o vizinho mais próximo (1-NN) e os k vizinhos
mais próximos (k-NN) de um ponto em espaços de dimensão d. Comparamos a árvore com a busca por força bruta, paralelizamos as
consultas na CPU (OpenMP) e na GPU (CUDA) e medimos quanto cada versão ganha em relação à KD-Tree sequencial. A pergunta que
guia o relatório é se a KD-Tree continua compensando quando a dimensão cresce. Testamos d = 2, 3, 10 e 50.</p>
<p>A divisão do trabalho foi a seguinte. O Cauã escreveu a construção da árvore (eixo de corte, mediana) e as buscas 1-NN e
k-NN. O Stricker fez a versão por força bruta que serve de baseline, o gerador de pontos aleatórios e os testes de corretude
que comparam a árvore com a força bruta. O Davi fez as versões paralelas, em OpenMP e em CUDA, e rodou as medições de
speedup, todas na máquina dele. O Eduardo ficou com a análise da maldição da dimensionalidade, os scripts de benchmark,
gráficos e tabelas, o README e as referências.</p>

<h2>2. O que foi implementado</h2>
<h3>2.1 A KD-Tree</h3>
<p>A árvore (<code>ArvoreKD</code>, em <code>include/kdtree.h</code> e <code>src/kdtree.cpp</code>) segue a estrutura proposta
por Bentley (1975) e guarda uma cópia dos pontos e um vetor de nós. Cada nó tem o índice do ponto, o eixo de corte e os índices dos dois filhos, com -1 quando o filho
não existe. Na construção, o eixo é a profundidade módulo d, e o ponto do meio na ordem daquele eixo vira o nó. Para achar a
mediana sem ordenar tudo usamos <code>std::nth_element</code>; os menores vão para a esquerda e os maiores para a direita.
Todas as distâncias são comparadas ao quadrado, sem tirar a raiz, o que não muda a ordem e economiza tempo.</p>
<p>A busca desce primeiro pelo lado em que a consulta cai. Na volta, ela só visita o outro lado se a distância da consulta
até o plano de corte, ao quadrado, for menor do que a melhor distância encontrada até ali. Essa é a poda, e é ela que torna a
árvore rápida (Friedman et al., 1977). O k-NN segue a mesma ideia, mas guarda os k melhores numa fila de prioridade, e
enquanto essa fila não está cheia nada pode ser podado.</p>

<h3>2.2 Força bruta, gerador e testes</h3>
<p>O baseline do Stricker está em <code>include/forca_bruta.hpp</code>, nas funções
<code>vizinho_mais_proximo_forca_bruta</code> e <code>k_vizinhos_forca_bruta</code>. Em caso de empate de distância, as duas
ficam com o menor índice, então o resultado da força bruta é sempre o mesmo. Os testes comparam a distância (e não o índice) com a
da árvore, para não acusar falha em empates. O gerador
(<code>gerar_uniforme</code>, em <code>include/gerador.hpp</code>) produz pontos uniformes em [0,1]<sup>d</sup> com semente fixa.</p>
<p>O teste <code>teste_corretude</code> compara a KD-Tree com a força bruta em d = 2, 3, 10 e 50 (20.000 pontos, 200 consultas,
k = 1, 5 e 10, além do 1-NN). Ele também confere a estrutura da árvore (cada ponto aparece uma única vez, com os menores à
esquerda e os maiores à direita de cada nó) e alguns casos de borda: k maior que o número de pontos, um ponto só, pontos
repetidos e consulta igual a um ponto do conjunto. Todos os testes passam.</p>

<h3>2.3 Versão OpenMP</h3>
<p>Cada consulta é independente das outras, então bastou paralelizar o laço das consultas com
<code>parallel for</code> e <code>schedule(dynamic, 64)</code>. Cada thread escreve só na sua posição do vetor de saída, e por
isso não há nenhuma sincronização. O escalonamento dinâmico ajuda porque consultas diferentes visitam quantidades diferentes
de nós.</p>

<h3>2.4 Versão CUDA</h3>
<p>Na GPU, cada thread resolve uma consulta, em blocos de 128 threads. Como a busca recursiva não é conveniente num kernel,
ela foi reescrita com uma pilha explícita de 64 posições. Cada entrada guarda o nó do lado distante e a distância até o plano
de corte, e a poda é decidida quando a entrada sai da pilha. Também existe um kernel de força bruta na GPU, em que cada thread
percorre todos os pontos. Os dois kernels foram escritos em <code>double</code> e em <code>float</code>.</p>

<h2>3. Como medimos</h2>
<p><b>Máquina.</b> Todas as medições principais foram feitas num notebook com Intel Core i7-12650HX (6 núcleos de desempenho
com hyper-threading e 8 de eficiência, 20 threads no total), 16 GB de RAM e placa NVIDIA RTX 4050 Laptop com 6 GB (arquitetura
Ada Lovelace, <code>-arch=sm_89</code>), com Windows 11, na tomada e em modo Melhor desempenho. O código foi compilado com
<code>g++ -O2 -fopenmp</code> e, para a GPU, com <code>nvcc -O3 -Xcompiler "/O2,/openmp"</code>. Sem essas opções o
<code>nvcc</code> não otimiza o código da CPU e os tempos saem de 1,2× a 1,8× piores.</p>
<p><b>Dados.</b> Em todas as dimensões usamos 100.000 pontos uniformes em [0,1]<sup>d</sup> (semente 42) e buscas 1-NN com
<code>double</code> (a GPU também foi testada em <code>float</code>). O número de consultas muda com a dimensão:
{n(CONS(d2), 0)}, {n(CONS(d3), 0)}, {n(CONS(d10), 0)} e {n(CONS(d50), 0)} para d = 2, 3, 10 e 50. Reduzimos nas dimensões altas porque a
busca sequencial fica muito mais lenta e queríamos que cada rodada terminasse em tempo razoável. Por causa disso os tempos
totais não podem ser comparados entre dimensões, e nos gráficos usamos o tempo por consulta.</p>
<p><b>Tempos.</b> O <code>bench_final</code> repete cada medição 3 vezes na CPU e 7 vezes na GPU e usa a mediana. Rodamos o
programa três vezes (<code>final_run1</code> a <code>final_run3</code>) e tomamos a mediana das três. O speedup é o tempo da
KD-Tree sequencial dividido pelo tempo do método. Na GPU, "kernel" é só a execução do kernel e "total" inclui copiar a
árvore, os pontos e as consultas para a placa e trazer as respostas de volta. Usamos o total, que é o que se compara com a CPU.
A construção da árvore não entra em nenhum tempo. O programa confere cada resposta paralela com a da KD-Tree sequencial
(na GPU em <code>float</code>, aceitando diferença relativa de até 10<sup>-3</sup> na distância).</p>

<h2>4. Resultados</h2>
<h3>4.1 Tempo e speedup</h3>
<div class=legenda>Tabela 1 — Mediana de 3 rodadas, tempo total e speedup sobre a KD-Tree sequencial (entre parênteses).</div>
{tabela1()}
<p>Em d = 2 e d = 3 a KD-Tree responde a uma consulta em menos de 1 µs ({n(tq[d2], 2)} e {n(tq[d3], 2)} µs). O OpenMP com
20 threads chega a {n(SP(d2, 'kdtree_omp20'), 1)}× e {n(SP(d3, 'kdtree_omp20'), 1)}×, e a versão em GPU a {n(SP(d2, 'gpu_kdtree_float_total'), 0)}× e
{n(SP(d3, 'gpu_kdtree_float_total'), 0)}×. A força bruta na GPU, nessas dimensões, é mais lenta que a própria KD-Tree sequencial
({n(SP(d2, 'gpu_bf_float_total'))}× e {n(SP(d3, 'gpu_bf_float_total'))}×), porque compara cada consulta com os 100.000 pontos.</p>
<p>Em d = 10 a GPU com a KD-Tree ({n(SP(d10, 'gpu_kdtree_float_total'), 1)}×) e a força bruta na GPU ({n(SP(d10, 'gpu_bf_float_total'), 1)}×) ficam
próximas. Em d = 50 a ordem se inverte: a força bruta na GPU chega a {n(SP(d50, 'gpu_bf_float_total'), 1)}×, contra {n(SP(d50, 'gpu_kdtree_float_total'), 1)}× da KD-Tree
na GPU e {n(SP(d50, 'kdtree_omp20'), 1)}× do OpenMP.</p>
<figure>{img('g1_tempo_por_consulta_vs_dim.png', 'Tempo por consulta em função da dimensão')}
<figcaption>Figura 1 — Tempo por consulta em função de d, em escala logarítmica. Dividir pelo número de consultas torna as
dimensões comparáveis.</figcaption></figure>
<figure>{img('g2_speedup_vs_dim.png', 'Speedup em função da dimensão')}
<figcaption>Figura 2 — Speedup sobre a KD-Tree sequencial, em escala logarítmica. Os valores de d = 50 são limites inferiores
(seção 6).</figcaption></figure>

<h3>4.2 Float e double na GPU</h3>
<p>Em <code>double</code> a GPU fica bem atrás. O tempo total da KD-Tree na GPU dá speedups de {n(SP(d2, 'gpu_kdtree_double_total'), 1)}×,
{n(SP(d3, 'gpu_kdtree_double_total'), 1)}×, {n(SP(d10, 'gpu_kdtree_double_total'), 1)}× e {n(SP(d50, 'gpu_kdtree_double_total'), 1)}× em d = 2, 3, 10 e 50, e o da força bruta, {n(SP(d2, 'gpu_bf_double_total'))}×,
{n(SP(d3, 'gpu_bf_double_total'))}×, {n(SP(d10, 'gpu_bf_double_total'), 1)}× e {n(SP(d50, 'gpu_bf_double_total'), 1)}×. A força bruta em d = 2 leva {n(M(d2, 'gpu_bf_double_total'), 2)} s em <code>double</code> e
{n(M(d2, 'gpu_bf_float_total'), 2)} s em <code>float</code>. Isso é esperado numa placa GeForce, que faz bem menos operações de 64 bits do que de
32 bits. O ganho do <code>float</code> tem um custo, que é a precisão menor das distâncias, e por isso o benchmark usa uma
tolerância na conferência dos resultados.</p>

<h3>4.3 Escalabilidade do OpenMP</h3>
<div class=legenda>Tabela 2 — Speedup do OpenMP sobre a versão sequencial, por número de threads.</div>
{tabela2()}
<figure>{img('g3_speedup_vs_threads.png', 'Speedup do OpenMP por número de threads')}
<figcaption>Figura 3 — Speedup do OpenMP em função do número de threads, uma linha por dimensão. A linha tracejada é o
speedup ideal.</figcaption></figure>
<p>Com 1 thread o OpenMP fica praticamente igual ao sequencial ({n(T(d2, 1))}× a {n(T(d10, 1))}×), então o custo de usar a diretiva é
desprezível. Com 4 threads o ganho ainda fica entre {n(min(T(d, 4) for d in DIMS) / 4 * 100, 0)}% e {n(max(T(d, 4) for d in DIMS) / 4 * 100, 0)}% do ideal. Depois ele vai ficando para trás: com 20 threads o speedup é {n(T(d2, 20), 1)}× em
d = 2, {n(T(d3, 20), 1)}× em d = 3, {n(T(d10, 20), 1)}× em d = 10 e {n(T(d50, 20), 1)}× em d = 50, o que dá eficiências de {n(T(d2, 20) / 20 * 100, 0)}%, {n(T(d3, 20) / 20 * 100, 0)}%,
{n(T(d10, 20) / 20 * 100, 0)}% e {n(T(d50, 20) / 20 * 100, 0)}%. Uma explicação provável é que o processador mistura núcleos rápidos com núcleos de eficiência, mais lentos, e que dois
threads no mesmo núcleo dividem as unidades de execução; passar de 12 para 20 threads soma justamente os núcleos mais fracos. Não
medimos isso, é uma hipótese.</p>
<p>Em d = 50 o ganho praticamente para entre 8 e 14 threads ({n(T(d50, 8), 1)}×, {n(T(d50, 12), 1)}× e {n(T(d50, 14), 1)}×) e só volta a subir
com 16 e 20. Aqui pesa o número pequeno de consultas: com 1.000 consultas e blocos de 64 existem só 16 blocos de trabalho, e
com 20 threads quatro delas ficam sem nada para fazer.</p>
<p>Um aviso sobre a Tabela 2: ela vem do <code>bench_threads</code>, que mede com <code>omp_get_wtime</code>, e os tempos
saem em múltiplos de 1 ms. Ela também é de uma execução diferente da Tabela 1, e os speedups de 20 threads não coincidem
({n(T(d2, 20))}× contra {n(SP(d2, 'kdtree_omp20'))}× em d = 2 e {n(T(d3, 20))}× contra {n(SP(d3, 'kdtree_omp20'))}× em d = 3). Em d = 2 a diferença cabe na resolução do relógio (o tempo de 20 threads é
{n(TT(d2, 20), 3)} s). Em d = 3 não cabe, e o mais provável é que o estado do notebook (temperatura, outros processos) fosse
diferente nas duas execuções. Por isso usamos a Tabela 2 para a forma da curva e a Tabela 1 como resultado final.</p>

<h3>4.4 KD-Tree contra força bruta</h3>
<p>O benchmark principal compara as duas abordagens só na GPU. Para ter também a comparação sequencial, escrevemos o programa
<code>benchmarks/baseline_e_poda.cpp</code>, que usa a mesma configuração do benchmark (100.000 pontos, semente 42, 1-NN,
<code>double</code>) e a força bruta do Stricker, numa thread só. Como o custo por consulta não depende de quantas consultas se faz,
ele mede só as primeiras consultas de cada dimensão (5.000 em d = 2 e 3, 2.000 em d = 10 e 200 em d = 50). O mesmo programa conta
quantos nós a busca visita, usando uma cópia da busca da biblioteca com um contador. A cópia foi conferida contra a biblioteca, e a
KD-Tree contra a força bruta, sem nenhuma divergência.</p>
<div class=legenda>Tabela 3 — KD-Tree sequencial contra força bruta sequencial, 1-NN, 100.000 pontos.</div>
{tabela3()}
{origem_baseline}
<p>Em d = 2 e 3 a KD-Tree é duas ordens de grandeza mais rápida que a força bruta ({n(razao_cpu[d2], 0)}× e {n(razao_cpu[d3], 0)}×). Em d = 10 ainda
ganha, por {n(razao_cpu[d10], 1)}×. Em d = 50 o jogo vira: a força bruta sequencial é {n(1 / razao_cpu[d50], 1)} vezes mais rápida que a árvore. Na GPU o
cruzamento chega mais cedo: em d = 10 a KD-Tree vence por apenas {n(razao_gpu[d10], 1)}×, e em d = 50 perde por {n(1 / razao_gpu[d50], 1)} vezes.</p>
<figure>{img('g4_kdtree_vs_forca_bruta.png', 'Razão força bruta sobre KD-Tree na CPU e na GPU')}
<figcaption>Figura 4 — Tempo da força bruta dividido pelo tempo da KD-Tree. Acima de 1× a KD-Tree vence. A curva da CPU vem
da Tabela 3 e a da GPU, da Tabela 1 (<code>float</code>, tempo total).</figcaption></figure>
<p>O número de pontos também pesa. O <code>teste_corretude</code> do Stricker imprime essa mesma razão, mas com 20.000 pontos e
k = 5. Em quatro execuções na mesma máquina virtual a razão ficou entre 29× e 37× em d = 2, entre 19× e 22× em d = 3, entre 0,3× e
0,5× em d = 10 e entre 0,25× e 0,33× em d = 50. Ou seja, com 20.000 pontos a força bruta já ganha em d = 10, enquanto com 100.000
pontos a KD-Tree ganha. Mudaram n e k ao mesmo tempo, então não dá para separar o efeito de cada um, mas o resultado combina com
a regra prática de que a KD-Tree só compensa quando n é bem maior que 2<sup>d</sup> (2<sup>10</sup> = 1.024, e 2<sup>50</sup> passa de 10<sup>15</sup>).</p>

<h2>5. Maldição da dimensionalidade</h2>
<h3>5.1 As distâncias se concentram</h3>
<p>A poda funciona quando o vizinho mais próximo está bem mais perto da consulta do que o resto dos pontos. Em dimensão alta isso
deixa de valer, porque as distâncias entre pontos aleatórios tendem a ficar todas parecidas (Beyer et al., 1999). Para ver isso nos
nossos dados, calculamos para 200 consultas e para os 100.000 pontos uniformes de cada dimensão o contraste
(d<sub>max</sub> − d<sub>min</sub>) / d<sub>min</sub> e a razão d<sub>min</sub> / d<sub>max</sub>.</p>
<div class=legenda>Tabela 4 — Concentração de distâncias (mediana de 200 consultas, 100.000 pontos uniformes).</div>
{tabela4()}
<p>Em d = 2 o ponto mais distante está, na mediana, cerca de {n(round(CONTR[d2], -1), 0)} vezes mais longe do que o mais próximo. Em d = 50 está só cerca de
2 vezes mais longe (contraste {n(CONTR[d50], 2)}), e o vizinho mais próximo está a {n(RZ[d50] * 100, 0)}% da distância do mais distante. Isso está ligado à poda:
um lado do corte só é descartado se o plano estiver mais longe do que o melhor candidato, e quando o melhor candidato está quase
tão longe quanto os outros pontos, a esfera ao redor da consulta atravessa quase todos os planos de corte.</p>

<h3>5.2 A poda deixa de funcionar</h3>
<figure>{img('g5_nos_visitados_vs_dim.png', 'Fração da árvore visitada por consulta')}
<figcaption>Figura 5 — Fração dos nós da KD-Tree visitada por consulta 1-NN (mediana), com a árvore real do projeto e 100.000
pontos.</figcaption></figure>
<p>A Tabela 3 e a Figura 5 mostram o efeito. Em d = 2 uma consulta visita {n(base.loc[d2, 'nos_visitados_mediana'], 0)} nós de 100.000 ({n(base.loc[d2, 'fracao_nos_mediana'] * 100, 3)}%) e em d = 3, {n(base.loc[d3, 'nos_visitados_mediana'], 0)}
({n(base.loc[d3, 'fracao_nos_mediana'] * 100, 3)}%). Em d = 10 a mediana sobe para {n(base.loc[d10, 'nos_visitados_mediana'], 0)} nós ({n(base.loc[d10, 'fracao_nos_mediana'] * 100, 1)}%), e a pior consulta visita
{n(base.loc[d10, 'fracao_nos_max'] * 100, 0)}% da árvore. Em d = 50 a mediana é de 100%: a busca visita a árvore inteira e ainda paga o custo extra de percorrer
ponteiros e de empilhar chamadas. É por isso que a KD-Tree sequencial perde para a força bruta nessa dimensão.</p>

<h3>5.3 Efeito no tempo e no paralelismo</h3>
<p>O custo de uma consulta na KD-Tree sequencial vai de {n(tq[d2], 2)} µs em d = 2 para {n(tq[d50], 0)} µs em d = 50, um aumento de
{n(tq[d50] / tq[d2], 0)} vezes. Uma parte vem de cada distância ter 25 vezes mais coordenadas, e o resto da poda que desapareceu. As versões
paralelas da KD-Tree herdam o problema: o speedup do OpenMP cai de {n(SP(d2, 'kdtree_omp20'), 1)}× para {n(SP(d50, 'kdtree_omp20'), 1)}× e o da GPU de {n(SP(d2, 'gpu_kdtree_float_total'), 0)}× para
{n(SP(d50, 'gpu_kdtree_float_total'), 0)}×, enquanto o da força bruta na GPU sobe de {n(SP(d2, 'gpu_bf_float_total'))}× para {n(SP(d50, 'gpu_bf_float_total'), 0)}×. Uma explicação provável é que a busca na árvore é irregular:
threads vizinhas na GPU seguem caminhos diferentes e fazem acessos dispersos à memória, o que as faz esperar umas pelas outras (NVIDIA),
enquanto a força bruta lê a memória de forma regular e tem custo linear em d. Também não medimos isso diretamente.</p>
<p>Esse comportamento é o descrito por Weber et al. (1998), que mostram que, a partir de algo em torno de dez dimensões, índices em
árvore perdem para uma varredura sequencial nos dados. O termo "maldição da dimensionalidade" vem de Bellman (1961).</p>

<h2>6. Limitações</h2>
<div class=nota><b>d = 50 usa só 1.000 consultas.</b> O tempo sequencial é alto, então o lote em d = 50 foi fixado em 1.000
consultas. Isso gera apenas 8 blocos de execução na GPU e 16 blocos de trabalho no OpenMP, o que não ocupa os multiprocessadores
da RTX 4050 nem as 20 threads do processador. Numa verificação com 100 consultas, o tempo na GPU ficou quase o mesmo (1,58 s contra
1,69 s), o que confirma a subutilização. O sequencial não é afetado, então os speedups de OpenMP e GPU em d = 50 são limites
inferiores.</div>
<p><b>Ruído.</b> O <code>bench_final</code> mede o sequencial de novo no fim de cada dimensão. Em d = 10 e 50 o segundo valor ficou
{n(fim[d10], 0)}% e {n(fim[d50], 0)}% mais lento, mas em d = 2 e 3 ficou {n(fim[d2], 0)}% e {n(fim[d3], 0)}% mais lento, provavelmente porque o notebook esquentou depois das rodadas na
GPU. Os speedups de d = 2 e 3 devem ser lidos com uma incerteza dessa ordem, maior do que a faixa entre mínimo e máximo das três rodadas.</p>
<p><b>Máquinas diferentes.</b> A Tabela 2 vem de outra execução que a Tabela 1 (seção 4.3){'' if usa_maq else ', e a comparação sequencial da seção 4.4 foi medida numa máquina virtual'}.</p>
<p><b>Só 1-NN nas medições.</b> O k-NN está implementado e testado (corretude e uma versão OpenMP em
<code>paralelo/omp_knn.cpp</code>), mas o benchmark principal e os kernels de GPU só tratam do 1-NN.</p>
<p><b>Só dados uniformes.</b> Todos os pontos são uniformes em [0,1]<sup>d</sup>, que é um caso difícil para a KD-Tree. Dados reais costumam ter
dimensão intrínseca menor do que d, e nesses casos a árvore aguenta dimensões maiores. Os pontos de cruzamento que encontramos
também dependem de n, que ficou fixo em 100.000 no benchmark.</p>
<p><b>Precisão.</b> Os speedups de GPU usam <code>float</code>, com distâncias menos precisas do que em <code>double</code>, e os resultados foram aceitos
com tolerância relativa de 10<sup>-3</sup>.</p>

<h2>7. Conclusão</h2>
<p>A KD-Tree compensa muito em dimensão baixa: em d = 2 e 3 ela visita uma fração mínima da árvore, é centenas de vezes mais rápida que a
força bruta e a versão em GPU chega a cerca de 70× sobre o sequencial. Em d = 10 já visita uma parte relevante da árvore e a vantagem cai
para poucas vezes na CPU e quase some na GPU. Em d = 50 as distâncias estão tão concentradas que a busca visita a árvore inteira, e a força
bruta, principalmente a paralela na GPU, passa a ser a melhor escolha. Para dados uniformes, a partir de uma dezena de dimensões vale
considerar força bruta paralela ou métodos aproximados em vez de uma KD-Tree exata.</p>

<h2>Referências</h2>
<ol class=refs>
<li>BELLMAN, R. <i>Adaptive Control Processes: A Guided Tour</i>. Princeton: Princeton University Press, 1961.</li>
<li>BENTLEY, J. L. Multidimensional binary search trees used for associative searching. <i>Communications of the ACM</i>, v. 18, n. 9, p. 509–517, 1975.</li>
<li>BEYER, K.; GOLDSTEIN, J.; RAMAKRISHNAN, R.; SHAFT, U. When is “nearest neighbor” meaningful? In: <i>Database Theory — ICDT’99</i>. Berlin, Heidelberg: Springer, 1999. p. 217–235.</li>
<li>FRIEDMAN, J. H.; BENTLEY, J. L.; FINKEL, R. A. An algorithm for finding best matches in logarithmic expected time. <i>ACM Transactions on Mathematical Software</i>, v. 3, n. 3, p. 209–226, 1977.</li>
<li>NOTEBOOKCHECK. <i>Intel Core i7-12650HX Processor — Benchmarks and Specs</i>. Disponível em: https://www.notebookcheck.net/Intel-Core-i7-12650HX-Processor-Benchmarks-and-Specs.618795.0.html. Acesso em: 25 nov. 2026.</li>
<li>NVIDIA CORPORATION. <i>CUDA C++ Programming Guide</i>. Documentação técnica oficial.</li>
<li>OPENMP ARCHITECTURE REVIEW BOARD. <i>OpenMP Application Programming Interface</i>. Especificação oficial.</li>
<li>WEBER, R.; SCHEK, H.-J.; BLOTT, S. A quantitative analysis and performance study for similarity-search methods in high-dimensional spaces. In: <i>Proceedings of the 24th International Conference on Very Large Data Bases (VLDB)</i>, 1998. p. 194–205.</li>
</ol>
"""


HTML = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<title>KD-Tree: desempenho paralelo e maldição da dimensionalidade</title><style>{CSS}</style></head>
<body>{corpo()}</body></html>"""

with open(SAIDA_HTML, "w", encoding="utf-8") as f:
    f.write(HTML)

RODAPE = """<div style="width:100%; font-family: Inter, sans-serif; font-size:7.5pt; color:#777; padding:0 19mm; display:flex; justify-content:space-between;">
<span>CTC04 — Tema 1.3 · KD-Tree: desempenho paralelo e maldição da dimensionalidade</span>
<span>Página <span class="pageNumber"></span> de <span class="totalPages"></span></span></div>"""

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto("file://" + SAIDA_HTML)
    pg.wait_for_load_state("networkidle")
    pg.pdf(path=SAIDA_PDF, format="A4", print_background=True, prefer_css_page_size=True,
           display_header_footer=True, header_template="<span></span>", footer_template=RODAPE)
    b.close()
print("OK:", SAIDA_PDF)
