#!/usr/bin/env python3
"""Gera graficos e tabelas do relatorio (Item 5 e benchmarks) a partir dos CSVs.

Uso (na raiz do projeto):
    python scripts/gerar_graficos.py

Le:     resultados/resultados_final_mediana.csv, resultados/resultados_threads.csv
Escreve: graficos/*.png, tabelas/*.csv e tabelas/*.md
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(RAIZ, "resultados")
GRAF = os.path.join(RAIZ, "graficos")
TAB = os.path.join(RAIZ, "tabelas")
os.makedirs(GRAF, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# Paleta categorica validada (slots 1-4, ordem fixa), superficie branca.
AZUL, LARANJA, AQUA, AMARELO = "#257de0", "#f47230", "#00b894", "#f2a600"
TEXTO, TEXTO2, GRADE, EIXO = "#191919", "#666666", "#e7e7e7", "#c2c2c2"

METODOS = [  # (nome no CSV, rotulo, cor, marcador)
    ("kdtree_seq", "KD-Tree sequencial", AZUL, "o"),
    ("kdtree_omp20", "KD-Tree OpenMP (20 threads)", LARANJA, "s"),
    ("gpu_kdtree_float_total", "KD-Tree GPU (float, total)", AQUA, "^"),
    ("gpu_bf_float_total", "Forca bruta GPU (float, total)", AMARELO, "D"),
]
DIMS = [2, 3, 10, 50]

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": EIXO, "axes.labelcolor": TEXTO2,
    "xtick.color": TEXTO2, "ytick.color": TEXTO2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 9,
})


def estilo(ax, titulo, subtitulo=None):
    ax.set_title(titulo, loc="left", color=TEXTO, fontsize=12, fontweight="bold",
                 pad=22 if subtitulo else 8)
    if subtitulo:
        ax.text(0, 1.02, subtitulo, transform=ax.transAxes, color=TEXTO2, fontsize=9)


def br(v):
    """Numero com virgula decimal e ponto de milhar (pt-BR), poucos algarismos."""
    if v >= 100:
        return f"{v:,.0f}".replace(",", ".")
    if v >= 10:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}".replace(".", ",")
    return f"{v:.2f}".replace(".", ",")


def rotulo_final(ax, x, y, texto, cor_marcador):
    ax.annotate(texto, (x, y), xytext=(8, 0), textcoords="offset points",
                color=TEXTO, fontsize=8, va="center")


med = pd.read_csv(os.path.join(RES, "resultados_final_mediana.csv"))
thr = pd.read_csv(os.path.join(RES, "resultados_threads.csv"))

# Consultas por dimensao (N de consultas muda com d: 200k, 200k, 20k, 1k).
consultas = med.groupby("dim")["consultas"].first().to_dict()
med["us_por_consulta"] = med["tempo_mediana_s"] / med["consultas"] * 1e6


def serie(metodo, coluna):
    s = med[med.metodo == metodo].set_index("dim")[coluna]
    return [s[d] for d in DIMS]


# ---------- Grafico 1: tempo por consulta x dimensao (log) ----------
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for nome, rot, cor, mk in METODOS:
    ys = serie(nome, "us_por_consulta")
    ax.plot(DIMS, ys, color=cor, marker=mk, markersize=7, linewidth=2,
            markeredgecolor="white", markeredgewidth=1.2, label=rot)
    rotulo_final(ax, DIMS[-1], ys[-1], f"{br(ys[-1])} µs", cor)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xticks(DIMS)
ax.set_xticklabels([str(d) for d in DIMS])
ax.minorticks_off()
ax.set_xlabel("dimensão d")
ax.set_ylabel("tempo por consulta (µs, escala log)")
ax.set_xlim(1.8, 85)
estilo(ax, "Tempo por consulta × dimensão",
       "Mediana de 3 rodadas, dividida pelo nº de consultas (200 mil, 200 mil, 20 mil e 1 mil).")
ax.set_xlim(1.8, 120)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(GRAF, "g1_tempo_por_consulta_vs_dim.png"), dpi=200)
plt.close(fig)

# ---------- Grafico 2: speedup x dimensao ----------
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for nome, rot, cor, mk in METODOS[1:]:
    ys = serie(nome, "speedup_vs_seq")
    ax.plot(DIMS, ys, color=cor, marker=mk, markersize=7, linewidth=2,
            markeredgecolor="white", markeredgewidth=1.2, label=rot)
    rotulo_final(ax, DIMS[-1], ys[-1], f"{br(ys[-1])}×", cor)
ax.axhline(1, color=EIXO, linewidth=1)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xticks(DIMS)
ax.set_xticklabels([str(d) for d in DIMS])
ax.minorticks_off()
ax.set_yticks([1, 10, 100])
ax.set_yticklabels(["1×", "10×", "100×"])
ax.set_xlabel("dimensão d")
ax.set_ylabel("speedup sobre a KD-Tree sequencial")
ax.set_xlim(1.8, 120)
ax.set_ylim(0.35, 250)
estilo(ax, "Speedup × dimensão",
       "Referência: KD-Tree sequencial = 1×. Em d = 50, os valores são limites inferiores.")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(GRAF, "g2_speedup_vs_dim.png"), dpi=200)
plt.close(fig)

# ---------- Grafico 3: speedup x threads (OpenMP) ----------
cores_dim = {2: AZUL, 3: LARANJA, 10: AQUA, 50: AMARELO}
marc_dim = {2: "o", 3: "s", 10: "^", 50: "D"}
omp = thr[thr.metodo == "omp"]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
tmax = int(omp.threads.max())
ax.plot([1, tmax], [1, tmax], color=EIXO, linewidth=1, linestyle="--")
ax.text(tmax * 0.62, tmax * 0.62 * 1.08, "ideal", color=TEXTO2, fontsize=8)
for d in DIMS:
    s = omp[omp.dim == d].sort_values("threads")
    ax.plot(s.threads, s.speedup_vs_seq, color=cores_dim[d], marker=marc_dim[d],
            markersize=6, linewidth=2, markeredgecolor="white",
            markeredgewidth=1.0, label=f"d = {d}")
    rotulo_final(ax, s.threads.iloc[-1], s.speedup_vs_seq.iloc[-1],
                 f"{br(s.speedup_vs_seq.iloc[-1])}×", cores_dim[d])
ax.set_xticks(sorted(omp.threads.unique()))
ax.set_xlabel("número de threads")
ax.set_ylabel("speedup sobre a versão sequencial")
ax.set_xlim(0, tmax + 3.5)
estilo(ax, "Speedup do OpenMP × número de threads",
       "i7-12650HX, até 20 threads. Uma linha por dimensão.")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(GRAF, "g3_speedup_vs_threads.png"), dpi=200)
plt.close(fig)

# ---------- Baseline sequencial (forca bruta na CPU) e nos visitados ----------
# baseline_e_poda.csv   = rodado na maquina de testes (benchmarks/baseline_e_poda.cpp)
# baseline_e_poda_nuvem.csv = a mesma medicao feita em outra maquina (usada se o anterior nao existir)
BASE_MAQ = os.path.join(RES, "baseline_e_poda.csv")
BASE_ALT = os.path.join(RES, "baseline_e_poda_nuvem.csv")
if os.path.exists(BASE_MAQ):
    base, ORIGEM = pd.read_csv(BASE_MAQ), "máquina de testes"
else:
    base, ORIGEM = pd.read_csv(BASE_ALT), "outra máquina (Xeon 2,1 GHz)"
base = base.set_index("dim")

kd = med[med.metodo == "gpu_kdtree_float_total"].set_index("dim")["tempo_mediana_s"]
bf = med[med.metodo == "gpu_bf_float_total"].set_index("dim")["tempo_mediana_s"]
razao = [bf[d] / kd[d] for d in DIMS]                    # GPU: > 1 a KD-Tree vence
razao_cpu = [float(base.loc[d, "bf_sobre_kd"]) for d in DIMS]  # CPU sequencial

# ---------- Grafico 4: forca bruta / KD-Tree, CPU sequencial e GPU ----------
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for ys, rot, cor, mk, desl in [(razao_cpu, "CPU sequencial", AZUL, "o", 9),
                               (razao, "GPU (float, total)", AQUA, "^", -15)]:
    ax.plot(DIMS, ys, color=cor, marker=mk, markersize=7, linewidth=2,
            markeredgecolor="white", markeredgewidth=1.2, label=rot)
    for d, r in zip(DIMS, ys):
        ax.annotate(f"{br(r)}×", (d, r), xytext=(0, desl), textcoords="offset points",
                    ha="center", color=TEXTO, fontsize=8)
ax.axhline(1, color=EIXO, linewidth=1)
ax.text(1.7, 1.12, "empate", color=TEXTO2, fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(DIMS)
ax.set_xticklabels([str(d) for d in DIMS])
ax.minorticks_off()
ax.set_yticks([0.1, 1, 10, 100])
ax.set_yticklabels(["0,1×", "1×", "10×", "100×"])
ax.set_xlabel("dimensão d")
ax.set_ylabel("tempo da força bruta ÷ tempo da KD-Tree")
ax.set_xlim(1.6, 70)
ax.set_ylim(0.08, 1000)
estilo(ax, "Quanto a KD-Tree ganha da força bruta",
       "Acima de 1× a KD-Tree vence; abaixo de 1×, a força bruta vence.")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(GRAF, "g4_kdtree_vs_forca_bruta.png"), dpi=200)
plt.close(fig)

# ---------- Grafico 5: fracao de nos visitados ----------
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ys = [float(base.loc[d, "fracao_nos_mediana"]) * 100 for d in DIMS]
ax.plot(DIMS, ys, color=AZUL, marker="o", markersize=7, linewidth=2,
        markeredgecolor="white", markeredgewidth=1.2)
for d, y in zip(DIMS, ys):
    txt = f"{y:.3f}".replace(".", ",") if y < 0.1 else br(y)
    ax.annotate(f"{txt}%", (d, y), xytext=(0, 9), textcoords="offset points",
                ha="center", color=TEXTO, fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(DIMS)
ax.set_xticklabels([str(d) for d in DIMS])
ax.minorticks_off()
ax.set_yticks([0.01, 0.1, 1, 10, 100])
ax.set_yticklabels(["0,01%", "0,1%", "1%", "10%", "100%"])
ax.set_ylim(0.007, 500)
ax.set_xlim(1.6, 70)
ax.set_xlabel("dimensão d")
ax.set_ylabel("nós visitados ÷ total de nós")
estilo(ax, "Fração da árvore visitada por consulta (1-NN)",
       "KD-Tree real do projeto, 100.000 pontos uniformes; mediana das consultas da amostra.")
fig.tight_layout()
fig.savefig(os.path.join(GRAF, "g5_nos_visitados_vs_dim.png"), dpi=200)
plt.close(fig)

# ---------- Tabelas ----------
def md(df, caminho):
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
        f.write("\n")


# T1: tempo e speedup por metodo e dimensao (mediana de 3 rodadas)
rows = []
for nome, rot, _, _ in METODOS:
    for d in DIMS:
        r = med[(med.metodo == nome) & (med.dim == d)].iloc[0]
        rows.append({
            "método": rot, "d": d, "consultas": int(r.consultas),
            "tempo mediana (s)": round(r.tempo_mediana_s, 6),
            "mín (s)": round(r.tempo_min_s, 6), "máx (s)": round(r.tempo_max_s, 6),
            "µs/consulta": round(r.us_por_consulta, 3),
            "speedup": round(r.speedup_vs_seq, 2),
        })
t1 = pd.DataFrame(rows)
t1.to_csv(os.path.join(TAB, "t1_tempo_speedup.csv"), index=False)
md(t1, os.path.join(TAB, "t1_tempo_speedup.md"))

# T2: speedup do OpenMP por threads
t2 = omp.pivot(index="threads", columns="dim", values="speedup_vs_seq").round(2)
t2.columns = [f"d={c}" for c in t2.columns]
t2 = t2.reset_index()
t2.to_csv(os.path.join(TAB, "t2_speedup_threads.csv"), index=False)
md(t2, os.path.join(TAB, "t2_speedup_threads.md"))

# T3: custo por consulta da KD-Tree sequencial relativo a d = 2
base_d2 = med[(med.metodo == "kdtree_seq") & (med.dim == 2)].us_por_consulta.iloc[0]
t3 = pd.DataFrame({
    "d": DIMS,
    "KD-Tree seq (µs/consulta)": [round(x, 2) for x in serie("kdtree_seq", "us_por_consulta")],
    "× d=2": [round(x / base_d2, 1) for x in serie("kdtree_seq", "us_por_consulta")],
    "KD-Tree GPU float (s)": [round(kd[d], 6) for d in DIMS],
    "Força bruta GPU float (s)": [round(bf[d], 6) for d in DIMS],
    "bruta ÷ KD (GPU)": [round(x, 2) for x in razao],
    "bruta ÷ KD (CPU seq.)": [round(x, 2) for x in razao_cpu],
    "fração de nós visitados (%)": [round(float(base.loc[d, "fracao_nos_mediana"]) * 100, 3) for d in DIMS],
})
t3.to_csv(os.path.join(TAB, "t3_maldicao.csv"), index=False)
md(t3, os.path.join(TAB, "t3_maldicao.md"))

print("OK: graficos em", GRAF, "| tabelas em", TAB)
