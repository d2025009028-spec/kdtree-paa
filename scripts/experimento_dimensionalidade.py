#!/usr/bin/env python3
"""Concentracao de distancias (apoio teorico do Item 5).

Em pontos uniformes em [0,1]^d (d = 2, 3, 10, 50; N = 100.000, como no benchmark), mede para
cada consulta o contraste (dmax - dmin) / dmin e a razao dmin/dmax (Beyer et al., 1999).
Independente do codigo C++. A fracao de nos visitados na KD-Tree vem de
benchmarks/baseline_e_poda.cpp (arvore real do projeto).

Uso: python scripts/experimento_dimensionalidade.py
Escreve: tabelas/t4_concentracao_distancias.csv/.md e graficos/g6_concentracao_distancias.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRAF, TAB = os.path.join(RAIZ, "graficos"), os.path.join(RAIZ, "tabelas")
os.makedirs(GRAF, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

SEED, N, Q = 7, 100000, 200
DIMS = [2, 3, 10, 50]
LARANJA, TEXTO, TEXTO2, GRADE, EIXO = "#f47230", "#191919", "#666666", "#e7e7e7", "#c2c2c2"
plt.rcParams.update({
    "font.family": "sans-serif", "axes.edgecolor": EIXO, "axes.labelcolor": TEXTO2,
    "xtick.color": TEXTO2, "ytick.color": TEXTO2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRADE,
    "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def br(v, casas=2):
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


rng = np.random.default_rng(SEED)
linhas = []
for d in DIMS:
    pts = rng.random((N, d))
    qs = rng.random((Q, d))
    contr, razao = [], []
    for q in qs:
        dist = np.sqrt(((pts - q) ** 2).sum(axis=1))
        contr.append((dist.max() - dist.min()) / dist.min())
        razao.append(dist.min() / dist.max())
    linhas.append({"d": d, "N": N, "consultas": Q,
                   "contraste mediano (dmax-dmin)/dmin": float(np.median(contr)),
                   "dmin/dmax mediano": float(np.median(razao))})
    print(linhas[-1], flush=True)
t4 = pd.DataFrame(linhas).round(4)
t4.to_csv(os.path.join(TAB, "t4_concentracao_distancias.csv"), index=False)
with open(os.path.join(TAB, "t4_concentracao_distancias.md"), "w", encoding="utf-8") as f:
    f.write(t4.to_markdown(index=False) + "\n")

fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.set_title("Concentração de distâncias", loc="left", color=TEXTO, fontsize=12,
             fontweight="bold", pad=22)
ax.text(0, 1.02, "Contraste (dmax − dmin) / dmin entre uma consulta e 100.000 pontos; mediana de 200 consultas.",
        transform=ax.transAxes, color=TEXTO2, fontsize=9)
ys = t4["contraste mediano (dmax-dmin)/dmin"]
ax.plot(DIMS, ys, color=LARANJA, marker="s", markersize=7, linewidth=2,
        markeredgecolor="white", markeredgewidth=1.2)
for d, y in zip(DIMS, ys):
    ax.annotate(br(y, 2 if y < 10 else 0), (d, y), xytext=(0, 9), textcoords="offset points",
                ha="center", color=TEXTO, fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(DIMS)
ax.set_xticklabels([str(d) for d in DIMS])
ax.minorticks_off()
ax.set_ylim(0.3, 20000)
ax.set_xlim(1.6, 70)
ax.set_xlabel("dimensão d")
ax.set_ylabel("contraste (escala log)")
fig.tight_layout()
fig.savefig(os.path.join(GRAF, "g6_concentracao_distancias.png"), dpi=200)
plt.close(fig)
print("OK")
