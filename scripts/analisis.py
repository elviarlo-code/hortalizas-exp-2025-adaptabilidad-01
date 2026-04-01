#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análisis descriptivo y exploratorio del experimento de hortalizas en sistema NFT.

Este script:
- Calcula estadísticas descriptivas globales y por tratamiento/cultivo.
- Ejecuta ANOVA de una vía como análisis exploratorio sobre submuestras.
- Calcula tamaños de efecto entre tratamientos.
- Genera ranking de tratamientos.
- Ejecuta PCA global y PCA por cultivo.
- Genera heatmap de variables estandarizadas.
- Genera boxplots por tratamiento.
- Guarda tablas, figuras y un reporte breve.

Importante:
- No modifica data/raw.
- El ANOVA se interpreta de manera exploratoria, no como inferencia fuerte.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import unicodedata

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==============================
# Configuración general
# ==============================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "raw" / "dataset_hortalizas_limpio.csv"

RESULTS_DIR = ROOT_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

TRAT_ORDER = ["T1", "T2", "T3", "T4"]

TRAT_LABELS = {
    "T1": "INIA",
    "T2": "La Molina",
    "T3": "FAO",
    "T4": "Hidroponika",
}

GROUP_COLS = ["cod_tratamiento", "tratamiento", "cultivo"]

REQUIRED_COLS = {
    "cod_tratamiento",
    "tratamiento",
    "cultivo",
    "altura_hojas_cm",
    "num_hojas",
    "diametro_cm",
    "longitud_raiz_cm",
    "peso_fresco_g",
}

sns.set_theme(style="whitegrid", context="talk")


# ==============================
# Utilidades
# ==============================
def ensure_directories() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower().replace(" ", "_").replace("/", "_")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLS.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    numeric_cols = [c for c in df.columns if c not in GROUP_COLS]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["cod_tratamiento"] = pd.Categorical(
        df["cod_tratamiento"],
        categories=TRAT_ORDER,
        ordered=True
    )

    return df


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in GROUP_COLS]


def group_label(row: pd.Series) -> str:
    cod = str(row["cod_tratamiento"])
    trat = TRAT_LABELS.get(cod, str(row["tratamiento"]))
    cult = str(row["cultivo"])
    return f"{trat} - {cult}"


# ==============================
# Estadísticas descriptivas
# ==============================
def descriptive_statistics(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    global_desc = df[numeric_cols].describe().T
    global_desc["cv_pct"] = (global_desc["std"] / global_desc["mean"]) * 100
    global_desc.to_csv(TABLES_DIR / "descriptive_statistics_global.csv", index=True)

    by_treatment = (
        df.groupby(["cod_tratamiento", "tratamiento"], observed=False)[numeric_cols]
        .agg(["count", "mean", "std", "min", "median", "max"])
    )
    by_treatment.to_csv(TABLES_DIR / "descriptive_statistics_by_treatment.csv")

    by_crop = (
        df.groupby(["cultivo"], observed=False)[numeric_cols]
        .agg(["count", "mean", "std", "min", "median", "max"])
    )
    by_crop.to_csv(TABLES_DIR / "descriptive_statistics_by_crop.csv")

    by_treatment_crop = (
        df.groupby(["cod_tratamiento", "tratamiento", "cultivo"], observed=False)[numeric_cols]
        .agg(["count", "mean", "std", "min", "median", "max"])
    )
    by_treatment_crop.to_csv(TABLES_DIR / "descriptive_statistics_by_treatment_crop.csv")


def treatment_ranking(df: pd.DataFrame) -> None:
    ranking = (
        df.groupby(["cod_tratamiento", "tratamiento"], observed=False)["peso_fresco_g"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    ranking["ranking_peso_fresco"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(TABLES_DIR / "treatment_ranking_peso_fresco.csv", index=False)


# ==============================
# ANOVA exploratorio
# ==============================
def one_way_anova_table(df: pd.DataFrame, response_col: str, group_col: str = "cod_tratamiento") -> dict[str, float]:
    clean = df[[group_col, response_col]].dropna()
    groups = [g[response_col].values for _, g in clean.groupby(group_col, observed=False)]

    groups = [g for g in groups if len(g) > 0]
    k = len(groups)
    n = sum(len(g) for g in groups)

    if k < 2 or n <= k:
        return {
            "n": n,
            "k_groups": k,
            "ss_between": np.nan,
            "ss_within": np.nan,
            "df_between": np.nan,
            "df_within": np.nan,
            "ms_between": np.nan,
            "ms_within": np.nan,
            "f_value": np.nan,
            "eta_sq": np.nan,
        }

    grand_mean = np.mean(np.concatenate(groups))
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)

    df_between = k - 1
    df_within = n - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    f_value = ms_between / ms_within if ms_within and ms_within > 0 else np.nan

    ss_total = ss_between + ss_within
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

    return {
        "n": n,
        "k_groups": k,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "df_between": df_between,
        "df_within": df_within,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "f_value": f_value,
        "eta_sq": eta_sq,
    }


def run_anova_exploratory(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    rows_global = []
    for col in numeric_cols:
        row = one_way_anova_table(df, response_col=col, group_col="cod_tratamiento")
        row.update({"variable": col, "scope": "global", "cultivo": "todos"})
        rows_global.append(row)

    out_global = pd.DataFrame(rows_global).sort_values("f_value", ascending=False)
    out_global.to_csv(TABLES_DIR / "anova_exploratory_global.csv", index=False)

    rows_crop = []
    for cultivo, sub_df in df.groupby("cultivo", observed=False):
        for col in numeric_cols:
            row = one_way_anova_table(sub_df, response_col=col, group_col="cod_tratamiento")
            row.update({"variable": col, "scope": "by_crop", "cultivo": cultivo})
            rows_crop.append(row)

    out_crop = pd.DataFrame(rows_crop).sort_values(["cultivo", "f_value"], ascending=[True, False])
    out_crop.to_csv(TABLES_DIR / "anova_exploratory_by_crop.csv", index=False)


# ==============================
# Tamaños de efecto
# ==============================
def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan

    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if pooled_sd == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled_sd


def hedges_g(d: float, nx: int, ny: int) -> float:
    if np.isnan(d):
        return np.nan
    dof = nx + ny - 2
    if dof <= 0:
        return np.nan
    correction = 1 - (3 / (4 * dof - 1))
    return d * correction


def effect_sizes(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    pairs = list(combinations(TRAT_ORDER, 2))
    rows = []

    for var in numeric_cols:
        for t1, t2 in pairs:
            x = df.loc[df["cod_tratamiento"] == t1, var].dropna().values
            y = df.loc[df["cod_tratamiento"] == t2, var].dropna().values
            d = cohens_d(x, y)
            g = hedges_g(d, len(x), len(y))
            rows.append({
                "scope": "global",
                "cultivo": "todos",
                "variable": var,
                "trat_1": t1,
                "trat_2": t2,
                "trat_1_label": TRAT_LABELS.get(t1, t1),
                "trat_2_label": TRAT_LABELS.get(t2, t2),
                "n_1": len(x),
                "n_2": len(y),
                "cohens_d": d,
                "hedges_g": g,
                "abs_hedges_g": abs(g) if not np.isnan(g) else np.nan,
            })

    for cultivo, sub_df in df.groupby("cultivo", observed=False):
        for var in numeric_cols:
            for t1, t2 in pairs:
                x = sub_df.loc[sub_df["cod_tratamiento"] == t1, var].dropna().values
                y = sub_df.loc[sub_df["cod_tratamiento"] == t2, var].dropna().values
                d = cohens_d(x, y)
                g = hedges_g(d, len(x), len(y))
                rows.append({
                    "scope": "by_crop",
                    "cultivo": cultivo,
                    "variable": var,
                    "trat_1": t1,
                    "trat_2": t2,
                    "trat_1_label": TRAT_LABELS.get(t1, t1),
                    "trat_2_label": TRAT_LABELS.get(t2, t2),
                    "n_1": len(x),
                    "n_2": len(y),
                    "cohens_d": d,
                    "hedges_g": g,
                    "abs_hedges_g": abs(g) if not np.isnan(g) else np.nan,
                })

    out = pd.DataFrame(rows).sort_values(
        ["scope", "cultivo", "variable", "abs_hedges_g"],
        ascending=[True, True, True, False]
    )
    out.to_csv(TABLES_DIR / "effect_sizes_between_treatments.csv", index=False)


# ==============================
# PCA
# ==============================
def save_pca_results(df: pd.DataFrame, numeric_cols: list[str], prefix: str, subtitle: str) -> None:
    clean = df.dropna(subset=numeric_cols).copy()
    if clean.shape[0] < 3:
        return

    X = clean[numeric_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(len(numeric_cols), X_scaled.shape[0]))
    scores = pca.fit_transform(X_scaled)

    scores_df = clean[["cod_tratamiento", "tratamiento", "cultivo"]].reset_index(drop=True)
    scores_df["PC1"] = scores[:, 0]
    scores_df["PC2"] = scores[:, 1] if scores.shape[1] > 1 else 0.0
    scores_df["group_label"] = scores_df.apply(group_label, axis=1)
    scores_df.to_csv(TABLES_DIR / f"pca_scores_{prefix}.csv", index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=numeric_cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    )
    loadings.to_csv(TABLES_DIR / f"pca_loadings_{prefix}.csv")

    explained = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(pca.n_components_)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })
    explained.to_csv(TABLES_DIR / f"pca_variance_{prefix}.csv", index=False)

    # Scatter PCA
    plt.figure(figsize=(11, 8))
    ax = sns.scatterplot(
        data=scores_df,
        x="PC1",
        y="PC2",
        hue="cod_tratamiento",
        style="cultivo",
        hue_order=TRAT_ORDER,
        s=90,
        alpha=0.85,
    )

    pc1_var = explained.loc[0, "explained_variance_ratio"] * 100
    pc2_var = explained.loc[1, "explained_variance_ratio"] * 100 if len(explained) > 1 else 0.0

    plt.title(f"PCA - {subtitle}")
    plt.xlabel(f"PC1 ({pc1_var:.1f}%)")
    plt.ylabel(f"PC2 ({pc2_var:.1f}%)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"pca_scatter_{prefix}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"pca_scatter_{prefix}.pdf")
    plt.close()

    # Biplot básico
    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        data=scores_df,
        x="PC1",
        y="PC2",
        hue="cod_tratamiento",
        style="cultivo",
        hue_order=TRAT_ORDER,
        s=80,
        alpha=0.75,
    )

    scaling_factor = max(
        np.max(np.abs(scores_df["PC1"])) if not scores_df.empty else 1,
        np.max(np.abs(scores_df["PC2"])) if not scores_df.empty else 1,
    ) * 0.35

    for var in numeric_cols:
        x = loadings.loc[var, "PC1"] * scaling_factor
        y = loadings.loc[var, "PC2"] * scaling_factor if "PC2" in loadings.columns else 0.0
        plt.arrow(0, 0, x, y, color="black", alpha=0.7, head_width=0.08, length_includes_head=True)
        plt.text(x * 1.08, y * 1.08, var, fontsize=10)

    plt.title(f"PCA Biplot - {subtitle}")
    plt.xlabel(f"PC1 ({pc1_var:.1f}%)")
    plt.ylabel(f"PC2 ({pc2_var:.1f}%)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"pca_biplot_{prefix}.png", dpi=300)
    plt.savefig(FIGURES_DIR / f"pca_biplot_{prefix}.pdf")
    plt.close()


# ==============================
# Visualizaciones
# ==============================
def standardized_heatmap(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    clean = df.dropna(subset=numeric_cols).copy()
    scaler = StandardScaler()
    Z = scaler.fit_transform(clean[numeric_cols])

    z_df = pd.DataFrame(Z, columns=numeric_cols, index=clean.index)
    z_df["group_label"] = clean.apply(group_label, axis=1)

    heat = z_df.groupby("group_label")[numeric_cols].mean().sort_index()

    plt.figure(figsize=(14, max(6, 0.45 * len(heat))))
    sns.heatmap(
        heat,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Valor estandarizado (z-score)"},
    )
    plt.title("Heatmap de variables estandarizadas por tratamiento y cultivo")
    plt.xlabel("Variables")
    plt.ylabel("Grupos")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "heatmap_standardized_variables.png", dpi=300)
    plt.savefig(FIGURES_DIR / "heatmap_standardized_variables.pdf")
    plt.close()


def boxplots_by_treatment(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    for col in numeric_cols:
        plt.figure(figsize=(12, 7))
        ax = sns.boxplot(
            data=df,
            x="cod_tratamiento",
            y=col,
            hue="cultivo",
            order=TRAT_ORDER,
            showfliers=True,
        )
        sns.stripplot(
            data=df,
            x="cod_tratamiento",
            y=col,
            hue="cultivo",
            order=TRAT_ORDER,
            dodge=True,
            color="black",
            alpha=0.35,
            size=3,
            ax=ax,
        )

        handles, labels = ax.get_legend_handles_labels()
        n = df["cultivo"].dropna().nunique()
        ax.legend(
            handles[:n],
            labels[:n],
            title="Cultivo",
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

        plt.title(f"Distribución de {col} por tratamiento")
        plt.xlabel("Tratamiento")
        plt.ylabel(col)
        plt.tight_layout()

        safe_col = slugify(col)
        plt.savefig(FIGURES_DIR / f"boxplot_{safe_col}.png", dpi=300)
        plt.savefig(FIGURES_DIR / f"boxplot_{safe_col}.pdf")
        plt.close()


# ==============================
# Tabla resumen para manuscrito
# ==============================
def manuscript_summary_table(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(["cod_tratamiento", "tratamiento", "cultivo"], observed=False)
        .agg(
            n=("peso_fresco_g", "count"),
            altura_media_cm=("altura_hojas_cm", "mean"),
            hojas_media=("num_hojas", "mean"),
            diametro_medio_cm=("diametro_cm", "mean"),
            longitud_raiz_media_cm=("longitud_raiz_cm", "mean"),
            peso_fresco_medio_g=("peso_fresco_g", "mean"),
        )
        .reset_index()
        .sort_values(["cod_tratamiento", "cultivo"])
    )
    summary.to_csv(TABLES_DIR / "manuscript_summary_table.csv", index=False)


# ==============================
# Reporte textual
# ==============================
def write_summary_report(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    txt_path = REPORTS_DIR / "analysis_summary.txt"

    lines = [
        "ANALISIS EXPLORATORIO DE HORTALIZAS EN SISTEMA NFT",
        "=" * 65,
        f"Observaciones totales: {df.shape[0]}",
        f"Numero de columnas: {df.shape[1]}",
        f"Variables numericas analizadas: {', '.join(numeric_cols)}",
        f"Numero de tratamientos: {df['cod_tratamiento'].nunique()}",
        f"Numero de cultivos: {df['cultivo'].nunique()}",
        "",
        "Nota metodologica:",
        "- El tratamiento fue aplicado a nivel de modulo sin replica fisica independiente.",
        "- Las plantas evaluadas se interpretan como submuestras.",
        "- El ANOVA se reporta solo con fines exploratorios/descriptivos.",
        "- El PCA y los tamanos de efecto apoyan la interpretacion de patrones.",
    ]

    txt_path.write_text("\n".join(lines), encoding="utf-8")


# ==============================
# Main
# ==============================
def main() -> None:
    ensure_directories()
    df = load_data(INPUT_PATH)
    numeric_cols = get_numeric_columns(df)

    descriptive_statistics(df, numeric_cols)
    treatment_ranking(df)
    manuscript_summary_table(df)
    run_anova_exploratory(df, numeric_cols)
    effect_sizes(df, numeric_cols)

    save_pca_results(df, numeric_cols, prefix="global", subtitle="Global")
    for cultivo, sub_df in df.groupby("cultivo", observed=False):
        save_pca_results(
            sub_df,
            numeric_cols,
            prefix=f"crop_{slugify(cultivo)}",
            subtitle=f"Cultivo: {cultivo}"
        )

    standardized_heatmap(df, numeric_cols)
    boxplots_by_treatment(df, numeric_cols)
    write_summary_report(df, numeric_cols)

    print("Analisis completado.")
    print(f"Resultados guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()