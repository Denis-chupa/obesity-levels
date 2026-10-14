"""Figures for the report: data overview, cleaning, ablations, errors, progress.

Run from the project root:  python report/make_figures.py
Inputs: artifacts/train_full.parquet and the CSV files in results/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)
METERS = ["электричество", "холодная вода", "пар", "горячая вода"]
COLORS = ["#2c5d94", "#2a9d6f", "#e08e45", "#c0392b"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

columns = ["building_id", "meter", "timestamp", "meter_reading", "site_id", "constant_run"]
train = pd.read_parquet("artifacts/train_full.parquet", columns=columns)
train["y"] = np.log1p(train["meter_reading"])

# 1. Данные: строки по счётчикам и распределение целевой.
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
share = train["meter"].value_counts(normalize=True).sort_index()
axes[0].bar(METERS, share.to_numpy(), color=COLORS)
axes[0].set_title("Доля строк по типу счётчика")
axes[0].tick_params(axis="x", rotation=20)
for i, v in enumerate(share):
    axes[0].text(i, v, f"{v:.0%}", ha="center", va="bottom")
for meter in range(4):
    axes[1].hist(train.loc[train["meter"] == meter, "y"].sample(300_000, random_state=1),
                 bins=60, histtype="step", color=COLORS[meter], label=METERS[meter], density=True)
axes[1].set(title="log1p(meter_reading) по счётчикам", xlabel="log1p(meter_reading)")
axes[1].legend(fontsize=8)
plt.tight_layout()
fig.savefig(OUT / "data_overview.png", dpi=150)
plt.close(fig)

# 2. Чистка: здание электричества с самыми длинными нулевыми сериями.
electricity = train[train["meter"] == 0]
flagged = electricity.groupby("building_id")["constant_run"].sum().sort_values(ascending=False)
building = int(flagged.index[3])
series = electricity[electricity["building_id"] == building].sort_values("timestamp")
fig, axes = plt.subplots(1, 2, figsize=(14, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
ok = series[~series["constant_run"]]
bad = series[series["constant_run"]]
axes[0].plot(series["timestamp"], series["meter_reading"], color="#bbbbbb", lw=0.6)
axes[0].scatter(ok["timestamp"], ok["meter_reading"], s=1.5, color="#2c5d94", label="в обучении")
axes[0].scatter(bad["timestamp"], bad["meter_reading"], s=5, color="#c0392b", label="серия ≥ 168 ч: исключена")
axes[0].set(title=f"Электричество, здание {building}: исключённые серии", ylabel="meter_reading")
axes[0].legend(fontsize=8, loc="upper right")
by_meter = train.groupby("meter")["constant_run"].mean()
axes[1].bar(METERS, by_meter.to_numpy() * 100, color=COLORS)
axes[1].set(title="Доля исключённых строк, %")
axes[1].tick_params(axis="x", rotation=20)
for i, v in enumerate(by_meter):
    axes[1].text(i, v * 100, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig.savefig(OUT / "cleaning.png", dpi=150)
plt.close(fig)

# 3. Абляции: признаки и пороги чистки.
ablation = pd.read_csv("results/feature_ablation.csv")
cleaning = pd.read_csv("results/cleaning_ablation.csv")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
RU_FEATURES = {
    "current": "текущие признаки",
    "+ temperature windows": "+ окна температуры",
    "+ windows + holidays": "+ окна + праздники",
    "+ windows + holidays + month": "+ окна + праздники + месяц",
    "+ windows + holidays + day of year": "+ окна + праздники + день года",
    "+ windows + holidays + month + day of year": "+ окна + праздники + месяц + день года",
}
RU_CLEANING = {"none": "без чистки"}
part = ablation[(ablation["scheme"] == "blocks of 4 months") & (ablation["model"] == "LightGBM 400 trees / 127 leaves")].copy()
part["features"] = part["features"].map(RU_FEATURES)
base = part["mean"].iloc[0]
axes[0].barh(part["features"].iloc[::-1], part["mean"].iloc[::-1] - base,
             color=["#2a9d6f" if v < 0 else "#c0392b" for v in (part["mean"].iloc[::-1] - base)])
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set_title("Признаки: изменение RMSE на блоках месяцев")
axes[0].set_xlabel(f"Δ RMSE к базе ({base:.3f}); меньше — лучше")
axes[0].tick_params(axis="y", labelsize=8)
c = cleaning.sort_values("mean", ascending=False).copy()
c["fit_exclusion"] = (
    c["fit_exclusion"].replace(RU_CLEANING)
    .str.replace("zeros of meter 0 + constant runs", "нули электричества + постоянные", regex=False)
    .str.replace("zeros of meter 0 only", "только нули электричества", regex=False)
    .str.replace("constant runs only", "только постоянные", regex=False)
)
axes[1].barh(c["fit_exclusion"], c["mean"], color=["#2a9d6f" if "только нули" in n else "#2c5d94" for n in c["fit_exclusion"]])
axes[1].set_xlim(1.10, 1.15)
axes[1].set_title("Чистка: RMSE, среднее по 4 месяцам")
axes[1].tick_params(axis="y", labelsize=8)
plt.tight_layout()
fig.savefig(OUT / "ablations.png", dpi=150)
plt.close(fig)

# 4. Ошибки: по счётчикам и корреляция ошибок.
errors = pd.read_csv("results/error_by_meter.csv", index_col=0)
corr = pd.read_csv("results/error_correlation.csv", index_col=0)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
shares = errors.loc["доля строк"]
errors = errors.drop(index="доля строк").rename(columns=lambda c: METERS[int(c.split()[1])])
errors.drop(index="lightgbm: site_id + primary_use + meter").plot.bar(ax=axes[0], color=COLORS, width=0.8)
axes[0].set_title("RMSE по типу счётчика (доли строк: " + ", ".join(f"{v:.0%}" for v in shares) + ")")
axes[0].tick_params(axis="x", rotation=25, labelsize=8)
axes[0].set_ylim(0, 2.1)
axes[0].legend(fontsize=8, ncol=4, loc="upper left")
short = [n.replace("lightgbm", "LGBM").replace("catboost", "CB").replace("site_id + primary_use + meter", "site+use+meter").replace("site_id + meter", "site+meter").replace("building_id + meter", "building+meter") for n in corr.index]
im = axes[1].imshow(corr.to_numpy(), cmap="YlGnBu", vmin=0.7, vmax=1.0)
axes[1].set_xticks(range(len(short)), short, fontsize=7, rotation=35, ha="right")
axes[1].set_yticks(range(len(short)), short, fontsize=7)
for i in range(len(short)):
    for j in range(len(short)):
        axes[1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
axes[1].set_title("Корреляция ошибок моделей стэка")
plt.colorbar(im, ax=axes[1], fraction=0.046)
plt.tight_layout()
fig.savefig(OUT / "errors.png", dpi=150)
plt.close(fig)

# 5. Ошибка растёт с удалением от обучающих данных: две версии решения.
groups = {
    "версия 3\n(отправка 1.338)": [1.003, 1.125, 1.338],
    "итоговая версия\n(блоки месяцев, месяц)": [1.047, 1.145, 1.381],
}
labels = ["локальная проверка\n(разные схемы)", "Kaggle Public", "Kaggle Private"]
fig, ax = plt.subplots(figsize=(7.5, 3.9))
width = 0.36
for i, (name, values) in enumerate(groups.items()):
    positions = np.arange(3) + (i - 0.5) * width
    ax.bar(positions, values, width, label=name, color=["#6f9fd0", "#c0392b"][i])
    for x, v in zip(positions, values):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(3), labels)
ax.set_ylim(0.9, 1.45)
ax.set_title("RMSE: проверка внутри 2016 года и Kaggle (2017–2018)")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
fig.savefig(OUT / "horizon.png", dpi=150)
plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.png")))
