from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def annual_counts(path, start_year, end_year):
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    print(f"Reading: {path}", flush=True)

    frame = pd.read_excel(
        path,
        usecols=["Year"],
        engine="openpyxl",
    )

    years = pd.to_numeric(frame["Year"], errors="coerce")
    years = years.loc[years.between(start_year, end_year)]

    return years.astype(int).value_counts().sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    inputs = {
        "Documents": root / "data/scopus/scopus_full.xlsx",
        "Patent_Cited": root / "data/lda/lens_from_scopus_scholar.xlsx",
        "Policy_Cited": root / "data/lda/overton_from_scopus_scholar.xlsx",
    }

    output_dir = (
        root / "output/scopus_research_landscape/figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    annual = {
        name: annual_counts(path, 2015, 2026)
        for name, path in inputs.items()
    }

    # Use the years present in the Scopus baseline.
    data = (
        pd.DataFrame({"Documents": annual["Documents"]})
        .join(annual["Patent_Cited"].rename("Patent_Cited"))
        .join(annual["Policy_Cited"].rename("Policy_Cited"))
        .fillna(0)
        .astype(int)
        .rename_axis("Year")
        .reset_index()
        .sort_values("Year")
        .reset_index(drop=True)
    )

    if len(data) < 3:
        raise ValueError("At least three publication years are required.")

    if (data["Documents"] <= 0).any():
        raise ValueError("Scopus publication counts must be positive.")

    data["Patent_Rate"] = (
        100 * data["Patent_Cited"] / data["Documents"]
    )
    data["Policy_Rate"] = (
        100 * data["Policy_Cited"] / data["Documents"]
    )

    years = data["Year"].to_numpy()
    x = np.arange(len(data))

    publications = data["Documents"].to_numpy()
    patent_counts = data["Patent_Cited"].to_numpy()
    policy_counts = data["Policy_Cited"].to_numpy()

    patent_rate = data["Patent_Rate"].to_numpy()
    policy_rate = data["Policy_Rate"].to_numpy()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax1 = plt.subplots(
        figsize=(14, 7.5),
        facecolor="white",
    )

    # ========================================================
    # ORIGINAL PUBLICATION BARS
    # ========================================================

    bars = ax1.bar(
        x,
        publications,
        width=0.58,
        color="C0",
        alpha=0.22,
        label="Scopus publications",
        zorder=1,
    )

    for bar, count in zip(bars, publications):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + publications.max() * 0.015,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            zorder=3,
        )

    # Separate internal scale preserves the original appearance.
    ax2 = ax1.twinx()

    # ========================================================
    # ORIGINAL SMOOTH CURVES
    # ========================================================

    x_smooth = np.linspace(x.min(), x.max(), 300)

    ax2.plot(
        x_smooth,
        make_interp_spline(x, patent_rate, k=2)(x_smooth),
        color="C0",
        linewidth=2.2,
        label="Patent-cited share",
        zorder=3,
    )

    ax2.plot(
        x_smooth,
        make_interp_spline(x, policy_rate, k=2)(x_smooth),
        color="C1",
        linewidth=2.2,
        linestyle="--",
        label="Policy-cited share",
        zorder=3,
    )

    # ========================================================
    # ORIGINAL DONUT MARKERS
    # ========================================================

    for rates, color, halo_size, outer_size, inner_size in [
        (patent_rate, "C0", 610, 430, 145),
        (policy_rate, "C1", 530, 350, 115),
    ]:
        ax2.scatter(
            x, rates,
            s=halo_size,
            facecolor="white",
            edgecolor="none",
            zorder=4,
        )

        ax2.scatter(
            x, rates,
            s=outer_size,
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
            zorder=5,
        )

        ax2.scatter(
            x, rates,
            s=inner_size,
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=6,
        )

    # ========================================================
    # ORIGINAL ANNOTATION POSITIONS
    # ========================================================

    rate_range = max(
        float(np.ptp(np.concatenate([patent_rate, policy_rate]))),
        1.0,
    )

    patent_offset = rate_range * 0.105
    policy_offset = rate_range * 0.105

    patent_multipliers = {
        2015: 0.70,
        2016: 1.55,
        2017: -1.05,
        2018: 1.00,
        2019: 1.00,
        2020: 1.00,
        2021: 1.00,
        2022: 1.00,
        2023: 1.65,
        2024: 1.00,
        2025: 1.00,
        2026: 1.00,
    }

    policy_multipliers = {
        2015: 1.20,
        2016: 1.00,
        2017: -1.50,
        2018: 1.10,
        2019: 1.00,
        2020: 0.60,
        2021: -1.20,
        2022: 0.80,
        2023: 1.15,
        2024: 1.20,
        2025: 0.80,
        2026: 1.00,
    }

    annotation_style = {
        "ha": "center",
        "fontsize": 8.8,
        "fontweight": "bold",
        "bbox": {
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.96,
        },
        "arrowprops": {
            "arrowstyle": "-",
            "linewidth": 0.9,
            "linestyle": "--",
            "shrinkA": 8,
            "shrinkB": 4,
        },
        "zorder": 7,
    }

    for i, year in enumerate(years):
        multiplier = patent_multipliers.get(int(year), 1.0)

        ax2.annotate(
            f"Patent-cited: {patent_counts[i]:,}\n"
            f"Share: {patent_rate[i]:.2f}%",
            xy=(x[i], patent_rate[i]),
            xytext=(
                x[i],
                patent_rate[i] + patent_offset * multiplier,
            ),
            va="bottom" if multiplier >= 0 else "top",
            **annotation_style,
        )

        ax2.annotate(
            f"Policy-cited: {policy_counts[i]:,}\n"
            f"Share: {policy_rate[i]:.2f}%",
            xy=(x[i], policy_rate[i]),
            xytext=(
                x[i],
                policy_rate[i]
                - policy_offset * policy_multipliers.get(int(year), 1.0),
            ),
            va="top",
            **annotation_style,
        )

    # ========================================================
    # KEEP THE ORIGINAL SCALES AND HORIZONTAL AXIS
    # ========================================================

    ax1.set_ylim(0, publications.max() * 1.10)

    ax2.set_ylim(
        min(
            -rate_range * 0.18,
            policy_rate.min() - policy_offset * 1.6,
        ),
        patent_rate.max() + patent_offset * 1.8,
    )

    ax1.set_xlabel(
        "Publication year",
        fontsize=13,
        fontweight="bold",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        years,
        fontsize=10,
        fontweight="bold",
    )

    ax1.grid(
        axis="y",
        linestyle=":",
        alpha=0.40,
        zorder=0,
    )

    # ========================================================
    # HIDE BOTH VERTICAL AXES
    # Remove their titles, ticks, numbers, and vertical borders.
    # ========================================================

    for ax in (ax1, ax2):
        ax.set_ylabel("")

        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
        )

        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # ========================================================
    # ORIGINAL COMBINED LEGEND
    # ========================================================

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="lower left",
        bbox_to_anchor=(0.00, 0.10),
        frameon=True,
        fontsize=10,
    )

    fig.tight_layout()

    output_file = (
        output_dir / "academic_growth_patent_policy_annotated.pdf"
    )

    fig.savefig(
        output_file,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"Saved: {output_file}", flush=True)


if __name__ == "__main__":
    main()