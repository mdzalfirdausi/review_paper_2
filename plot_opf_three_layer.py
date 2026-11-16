from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.cm import ScalarMappable


def read_matrix(path, ncols):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Topic"] = pd.to_numeric(df["Topic"], errors="raise")
    df = df.set_index("Topic")

    if df.index.duplicated().any():
        raise ValueError(f"Duplicate topic IDs: {path}")

    expected_rows = list(range(1, 19))
    expected_cols = [str(i) for i in range(1, ncols + 1)]

    if set(df.index) != set(expected_rows):
        raise ValueError(f"Expected Scopus topic IDs 1–18: {path}")
    if set(df.columns) != set(expected_cols):
        raise ValueError(f"Unexpected external topic IDs: {path}")

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
        raise ValueError(f"Invalid overlap counts: {path}")

    return values.astype(int)


def draw_ribbon(ax, x0, x1, y0, y1, thickness, color):
    bend = 0.48 * (x1 - x0)

    vertices = [
        (x0, y0),
        (x0 + bend, y0),
        (x1 - bend, y1),
        (x1, y1),
        (x1, y1 + thickness),
        (x1 - bend, y1 + thickness),
        (x0 + bend, y0 + thickness),
        (x0, y0 + thickness),
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]

    ax.add_patch(
        PathPatch(
            MplPath(vertices, codes),
            facecolor=color,
            edgecolor="#D2D1D1",
            linewidth=0.35,
            alpha=0.75,
            zorder=1,
        )
    )


def draw_flows(
    ax, matrix, source_x, source_y, target_x, target_y,
    block_width, scale, colors, color_axis
):
    source_bottom = source_y - matrix.sum(axis=1) * scale / 2
    target_bottom = target_y - matrix.sum(axis=0) * scale / 2

    source_used = np.zeros(matrix.shape[0])
    target_used = np.zeros(matrix.shape[1])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = matrix[i, j]
            if count == 0:
                continue

            thickness = count * scale
            color = colors[i if color_axis == "source" else j]

            draw_ribbon(
                ax,
                source_x[i] + block_width / 2,
                target_x[j] - block_width / 2,
                source_bottom[i] + source_used[i],
                target_bottom[j] + target_used[j],
                thickness,
                color,
            )
            source_used[i] += thickness
            target_used[j] += thickness


def draw_blocks(ax, xs, ys, prefix, colors, width, height):
    for i, (x, y, color) in enumerate(zip(xs, ys, colors), start=1):
        ax.add_patch(
            Rectangle(
                (x - width / 2, y - height / 2),
                width,
                height,
                facecolor=color,
                edgecolor="white",
                linewidth=1.1,
                zorder=3,
            )
        )

        # Choose contrasting text for each block colour.
        r, g, b = color[:3]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        text_color = "#172B3A" if luminance > 0.60 else "white"

        ax.text(
            x, y, f"{prefix}{i}",
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color=text_color,
            zorder=4,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("figs"))
    args = parser.parse_args()

    patent = read_matrix(args.data_dir / "patent_flows.csv", 16)
    policy = read_matrix(args.data_dir / "policy_flows.csv", 9)

    patent_by_scopus = patent.sum(axis=1)
    policy_by_scopus = policy.sum(axis=1)
    combined = patent_by_scopus + policy_by_scopus

    # 0 = patent only; 0.5 = equal counts; 1 = policy only.
    # These are separate overlaps, not mutually exclusive publications.
    policy_share = np.divide(
        policy_by_scopus,
        combined,
        out=np.full(18, 0.5),
        where=combined > 0,
    )

    cmap = LinearSegmentedColormap.from_list(
        "patent_policy",
        ["#B96918", "#E9E6E2", "#61418B"],
    )
    scopus_colors = [cmap(float(value)) for value in policy_share]
    lens_colors = [matplotlib.colors.to_rgba("#B96918")] * 16
    overton_colors = [matplotlib.colors.to_rgba("#61418B")] * 9

    # Preserve topic-ID order vertically.
    y_scopus = np.linspace(0.93, 0.07, 18)
    y_lens = np.linspace(0.93, 0.07, 16)
    y_overton = np.linspace(0.93, 0.07, 9)

    x_lens = np.full(16, 0.08)
    x_overton = np.full(9, 0.92)

    # Patent-oriented topics shift left; policy-oriented topics shift right.
    x_scopus = 0.50 + 0.15 * (2 * policy_share - 1)

    block_width = 0.065
    block_height = 0.037

    max_bundle = max(
        patent.sum(axis=0).max(),
        patent_by_scopus.max(),
        policy.sum(axis=0).max(),
        policy_by_scopus.max(),
    )
    scale = (
        0.85 * block_height / max_bundle
        if max_bundle > 0 else 0
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(11.5, 10))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.025, 0.975)
    ax.axis("off")

    # Lens -> Scopus
    draw_flows(
        ax, patent.T,
        x_lens, y_lens, x_scopus, y_scopus,
        block_width, scale,
        scopus_colors, color_axis="target",
    )

    # Scopus -> Overton
    draw_flows(
        ax, policy,
        x_scopus, y_scopus, x_overton, y_overton,
        block_width, scale,
        scopus_colors, color_axis="source",
    )

    draw_blocks(
        ax, x_lens, y_lens, "L",
        lens_colors, block_width, block_height,
    )
    draw_blocks(
        ax, x_scopus, y_scopus, "S",
        scopus_colors, block_width, block_height,
    )
    draw_blocks(
        ax, x_overton, y_overton, "O",
        overton_colors, block_width, block_height,
    )

    for x, title in [
        (0.08, "Lens\nPatent-cited"),
        (0.50, "Scopus\nResearch topics"),
        (0.92, "Overton\nPolicy-cited"),
    ]:
        ax.text(
            x, 1.02, title,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
            color="#203040",
        )

    # Shared colour scale also explains horizontal displacement.
    cax = fig.add_axes([0.30, 0.067, 0.40, 0.014])
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
        cax=cax, orientation="horizontal",
    )
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["Patent", "Equal", "Policy"])
    colorbar.ax.tick_params(labelsize=9, length=0)
    colorbar.outline.set_visible(False)

    fig.text(
        0.5, 0.023,
        "Scopus position and colour reflect policy overlap / "
        "(patent overlap + policy overlap).\n"
        "Ribbon thickness represents publication overlap; "
        "blocks have equal sizes.",
        ha="center", fontsize=8.5, color="#52606D",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / "opf_three_layer_blocks"

    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Patent overlaps: {patent.sum():,}")
    print(f"Policy overlaps: {policy.sum():,}")
    print(f"Saved: {stem.with_suffix('.pdf')}")
    print(f"Saved: {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()