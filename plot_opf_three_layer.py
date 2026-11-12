from pathlib import Path
import argparse
import textwrap

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, FancyBboxPatch


# ------------------------------------------------------------
# Topic names: keep these consistent with the manuscript.
# List positions correspond to topic IDs, beginning at 1.
# ------------------------------------------------------------
SCOPUS = [
    "Quantum and quantum-inspired computing for power-system optimisation",
    "Blockchain and privacy-preserving optimisation of distributed energy resources",
    "General-purpose swarm and metaheuristic optimisation algorithms",
    "Reinforcement learning for power-system control and operation",
    "Neural and optimisation-based engineering modelling and design",
    "AI and digital technologies for smart-energy management",
    "ML-based load and renewable-generation forecasting",
    "Power-grid resilience, cascading failures, and outage risk",
    "Cyberattack and false-data detection in smart grids",
    "Graph and physics-informed neural learning for OPF",
    "Data-driven grid monitoring, fault diagnosis, and state estimation",
    "Electric-vehicle charging optimisation and vehicle–grid integration",
    "Microgrid and converter control for voltage and frequency stability",
    "Stochastic and robust optimisation of power-system operation",
    "Storage and renewable-energy system optimisation and management",
    "ADMM-based distributed optimisation and peer-to-peer energy coordination",
    "Distributed-generation allocation for loss reduction and voltage improvement",
    "Evolutionary optimisation of OPF, economic dispatch, and FACTS placement",
]

LENS = [
    "Smart-grid digitalisation, IoT, and digital twins",
    "Swarm and metaheuristic optimisation algorithms",
    "Evolutionary optimisation of DG, FACTS, and reactive power",
    "Electric-vehicle charging and battery energy management",
    "Deep reinforcement learning for grid control and operation",
    "Economic dispatch with renewable generation and storage",
    "ML-based load and renewable-generation forecasting",
    "Voltage and frequency control in renewable-rich grids",
    "Data-driven grid security assessment and operating limits",
    "Deep-learning fault diagnosis and equipment inspection",
    "Privacy-preserving distributed optimisation of integrated energy systems",
    "Physics-informed learning for grid event detection and state estimation",
    "Graph neural networks for topology-aware OPF",
    "Deep neural surrogates for AC and DC OPF",
    "Peer-to-peer energy trading and prosumer coordination",
    "ADMM-based distributed OPF and reactive-power control",
]

OVERTON = [
    "Smart electric-vehicle charging and vehicle–grid integration",
    "Grid resilience to cascading failures and cyberattacks",
    "Learning-based distribution-system state estimation",
    "AI and data-driven management of smart-energy systems",
    "Mathematical and distributed optimisation for OPF",
    "Microgrid supervisory control and energy-management optimisation",
    "Peer-to-peer electricity markets and trading coordination",
    "Uncertainty-aware forecasting, dispatch, and OPF",
    "Renewable-grid planning, storage, and market flexibility",
]

SCOPUS_COUNTS = [
    184, 232, 1871, 1247, 631, 686, 962, 325, 383,
    930, 1014, 766, 775, 774, 1025, 1597, 502, 1000
]


def read_flows(filename, n_external):
    """Read rows as Scopus topics and columns as external topics."""
    df = pd.read_csv(filename, encoding="utf-8-sig")
    if "Topic" not in df:
        raise ValueError(f"{filename}: missing 'Topic' column.")

    df["Topic"] = pd.to_numeric(df["Topic"], errors="raise")
    if df["Topic"].duplicated().any():
        raise ValueError(f"{filename}: duplicate Scopus topic IDs.")

    df = df.set_index("Topic")
    expected_rows = list(range(1, 19))
    expected_cols = [str(i) for i in range(1, n_external + 1)]

    if set(df.index) != set(expected_rows):
        raise ValueError(f"{filename}: expected Scopus topic IDs 1–18.")
    if set(df.columns) != set(expected_cols):
        raise ValueError(
            f"{filename}: expected external topic columns 1–{n_external}."
        )

    values = (
        df.loc[expected_rows, expected_cols]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=float)
    )

    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError(f"{filename}: counts must be nonnegative integers.")

    return values.astype(int)


def ribbon(ax, x0, x1, y0, y1, height, color):
    """Draw a count-scaled ribbon with smooth curved boundaries."""
    bend = 0.48 * (x1 - x0)
    vertices = [
        (x0, y0),
        (x0 + bend, y0),
        (x1 - bend, y1),
        (x1, y1),
        (x1, y1 + height),
        (x1 - bend, y1 + height),
        (x0 + bend, y0 + height),
        (x0, y0 + height),
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(
            MplPath(vertices, codes),
            facecolor=color,
            edgecolor="none",
            alpha=0.40,
            zorder=1,
        )
    )


def draw_connections(ax, matrix, source_y, target_y,
                     x0, x1, scale, colors, color_by):
    """Use one common count-to-thickness scale for both pathways."""
    source_totals = matrix.sum(axis=1)
    target_totals = matrix.sum(axis=0)

    source_bottom = source_y - source_totals * scale / 2
    target_bottom = target_y - target_totals * scale / 2
    source_used = np.zeros(len(source_y))
    target_used = np.zeros(len(target_y))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = matrix[i, j]
            if count == 0:
                continue

            height = count * scale
            y0 = source_bottom[i] + source_used[i]
            y1 = target_bottom[j] + target_used[j]
            color = colors[i if color_by == "source" else j]

            ribbon(ax, x0, x1, y0, y1, height, color)
            source_used[i] += height
            target_used[j] += height


def draw_topic_boxes(ax, x0, x1, centers, labels, prefix,
                     totals, colors, height, denominator=None):
    for index, (y, label, count, color) in enumerate(
        zip(centers, labels, totals, colors), start=1
    ):
        ax.add_patch(
            FancyBboxPatch(
                (x0, y - height / 2),
                x1 - x0,
                height,
                boxstyle="round,pad=0.002,rounding_size=0.003",
                linewidth=0.8,
                edgecolor=color,
                facecolor="white",
                zorder=3,
            )
        )

        # Narrow colour strip identifies each topic.
        ax.add_patch(
            plt.Rectangle(
                (x0, y - height / 2),
                0.004,
                height,
                color=color,
                linewidth=0,
                zorder=4,
            )
        )

        count_text = f"n={int(count):,}"
        if denominator:
            count_text += f"; {100 * count / denominator:.1f}%"

        heading = f"{prefix}{index}. {label}"
        lines = textwrap.wrap(
            heading, width=48,
            break_long_words=False,
            break_on_hyphens=False,
        )
        text = "\n".join(lines) + f"\n({count_text})"

        ax.text(
            (x0 + x1) / 2 + 0.002,
            y,
            text,
            ha="center",
            va="center",
            fontsize=8.3,
            linespacing=1.05,
            color="#203040",
            zorder=5,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("figs"))
    args = parser.parse_args()

    patent = read_flows(args.data_dir / "patent_flows.csv", 16)
    policy = read_flows(args.data_dir / "policy_flows.csv", 9)

    # Left-to-middle matrix: Lens rows, Scopus columns.
    left = patent.T
    right = policy

    lens_counts = left.sum(axis=1)
    overton_counts = right.sum(axis=0)
    n_lens = int(left.sum())
    n_overton = int(right.sum())

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(23, 15.5))
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.055, top=0.94)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Topic boxes occupy three columns; ribbons run between them.
    x_left = (0.010, 0.270)
    x_middle = (0.370, 0.630)
    x_right = (0.730, 0.990)

    y_scopus = np.linspace(0.925, 0.065, 18)
    y_lens = np.linspace(0.925, 0.065, 16)
    y_overton = np.linspace(0.925, 0.065, 9)
    box_height = 0.041

    cmap = plt.get_cmap("tab20")
    scopus_colors = [cmap(i) for i in range(18)]
    lens_colors = ["#B76A1C"] * 16
    overton_colors = ["#654491"] * 9

    # All ribbons use the same scale, including across the two sides.
    max_bundle = max(
        left.sum(axis=1).max(),
        left.sum(axis=0).max(),
        right.sum(axis=1).max(),
        right.sum(axis=0).max(),
    )
    scale = 0.88 * box_height / max_bundle

    draw_connections(
        ax, left, y_lens, y_scopus,
        x_left[1], x_middle[0],
        scale, scopus_colors, color_by="target",
    )
    draw_connections(
        ax, right, y_scopus, y_overton,
        x_middle[1], x_right[0],
        scale, scopus_colors, color_by="source",
    )

    draw_topic_boxes(
        ax, *x_left, y_lens, LENS, "L",
        lens_counts, lens_colors, box_height, n_lens,
    )
    draw_topic_boxes(
        ax, *x_middle, y_scopus, SCOPUS, "S",
        SCOPUS_COUNTS, scopus_colors, box_height,
    )
    draw_topic_boxes(
        ax, *x_right, y_overton, OVERTON, "O",
        overton_counts, overton_colors, box_height, n_overton,
    )

    headings = [
        (
            np.mean(x_left),
            f"Patent-cited topics (Lens)\n"
            f"{n_lens:,} publications linked to Scopus topics"
        ),
        (
            np.mean(x_middle),
            f"Research topics (Scopus)\n"
            f"{sum(SCOPUS_COUNTS):,} publications"
        ),
        (
            np.mean(x_right),
            f"Policy-cited topics (Overton)\n"
            f"{n_overton:,} publications linked to Scopus topics"
        ),
    ]
    for x, heading in headings:
        ax.text(
            x, 0.985, heading,
            ha="center", va="top",
            fontsize=13, fontweight="bold",
            color="#172B3A",
        )

    fig.text(
        0.5, 0.025,
        "Ribbon thickness = publication overlap count; "
        "colour = Scopus topic. "
        "Topic boxes have equal heights for readability.\n"
        "Lens and Overton percentages refer to their respective linked "
        "publications; the two pathways are separate overlaps.",
        ha="center", va="center", fontsize=9, color="#4C5964",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / "opf_three_layer_updated"

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight", facecolor="white",
    )
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)

    print(f"Lens–Scopus links: {n_lens:,}")
    print(f"Scopus–Overton links: {n_overton:,}")
    print(f"Saved: {stem.with_suffix('.pdf')}")
    print(f"Saved: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()