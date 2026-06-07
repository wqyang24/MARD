import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# =====================================================
# 0. Global font size configuration
#    All font sizes are enlarged to meet publication-quality
#    visualization standards.
# =====================================================
TITLE_FONTSIZE = 26
TICK_FONTSIZE = 20
CBAR_LABEL_FONTSIZE = 20
CBAR_TICK_FONTSIZE = 18
LABEL_FONTSIZE = 20
LINEWIDTH = 3  # Line width for highlighted rectangles

# =====================================================
# 0.1 Custom colormap definition
#     A perceptually ordered colormap ranging from
#     light grey (low magnitude) to dark blue (high magnitude).
# =====================================================
grey_blue_cmap = LinearSegmentedColormap.from_list(
    "grey_blue",
    ["#f2f2f2", "#041f4a"]
)

# =====================================================
# 1. Input CSV file paths
#    Each file corresponds to one model scale.
# =====================================================
csv_paths = [
    "./l2_diff_base-qwen-2.5-math-1.5b_vs_distill-deepseek-r1-distill-qwen-1.5b.csv",
    "./l2_diff_base-qwen-2.5-math-7b_vs_distill-deepseek-r1-distill-qwen-7b.csv",
    "./l2_diff_base-qwen-2.5-14b_vs_distill-deepseek-r1-distill-qwen-14b.csv",
    "./l2_diff_base-qwen-2.5-32b_vs_distill-deepseek-r1-distill-qwen-32b.csv",
]

# Model titles shown in subplots
titles = [
    "Qwen-2.5-1.5B",
    "Qwen-2.5-7B",
    "Qwen-2.5-14B",
    "Qwen-2.5-32B"
]

# =====================================================
# 2. Heatmap data loading and preprocessing
#    Each heatmap represents layer-wise L2 norm differences
#    across projection modules.
# =====================================================
def load_heatmap(csv_path):
    df = pd.read_csv(csv_path)

    # Construct a unique identifier for each projection module
    df["projection"] = df["module"] + "/" + df["proj_name"]

    # Ensure a consistent ordering for visualization
    df = df.sort_values(by=["projection", "layer_id"])

    # Pivot the table into a matrix suitable for heatmap plotting
    return df.pivot(
        index="projection",
        columns="layer_id",
        values="l2_norm",
    )

heatmaps = [load_heatmap(p) for p in csv_paths]

# =====================================================
# 3. Unified color normalization
#    A shared (vmin, vmax) is used across all subplots
#    to enable direct visual comparison between models.
# =====================================================
vmin = min(h.values.min() for h in heatmaps)
vmax = max(h.values.max() for h in heatmaps)

# =====================================================
# 4. GridSpec-based layout configuration
#    The width of each subplot is proportional to the
#    number of layers in the corresponding model.
# =====================================================
num_layers_list = [h.shape[1] for h in heatmaps]
cbar_width = 2
width_ratios = num_layers_list + [cbar_width]

fig = plt.figure(figsize=(sum(num_layers_list) * 0.25 + 2, 6))
gs = gridspec.GridSpec(
    nrows=1,
    ncols=5,
    width_ratios=width_ratios,
    wspace=0.1
)

axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
cax = fig.add_subplot(gs[0, 4])

# =====================================================
# 5. Heatmap rendering and structural highlighting
#    Extreme patterns are emphasized to facilitate analysis.
# =====================================================
for i, (ax, heatmap, title) in enumerate(zip(axes, heatmaps, titles)):
    im = ax.imshow(
        heatmap.values,
        aspect="auto",
        cmap=grey_blue_cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    # Y-axis: projection modules
    if i == 0:
        ax.set_yticks(range(len(heatmap.index)))
        ax.set_yticklabels(heatmap.index, fontsize=TICK_FONTSIZE)
    else:
        ax.set_yticks(range(len(heatmap.index)))
        ax.set_yticklabels([])

    # X-axis: transformer layers
    num_layers = heatmap.shape[1]
    ax.set_xticks([0, num_layers - 1])
    ax.set_xticklabels(["1", f"{num_layers}"], fontsize=TICK_FONTSIZE)
    ax.set_xlabel("Layer", fontsize=LABEL_FONTSIZE)

    # -------------------------------------------------
    # Highlight the row with the largest mean L2 norm
    # difference across layers (red rectangle).
    # -------------------------------------------------
    row_means = heatmap.mean(axis=1)
    max_row_idx = row_means.idxmax()
    max_row_loc = heatmap.index.get_loc(max_row_idx)

    ax.add_patch(
        patches.Rectangle(
            (-0.5, max_row_loc - 0.5),
            num_layers,
            1,
            linewidth=LINEWIDTH,
            edgecolor="red",
            facecolor="none",
        )
    )

    # -------------------------------------------------
    # Highlight the column with the smallest mean L2 norm
    # difference across projection modules (blue rectangle).
    # -------------------------------------------------
    col_means = heatmap.mean(axis=0)
    min_col_idx = col_means.idxmin()
    min_col_loc = heatmap.columns.get_loc(min_col_idx)

    ax.add_patch(
        patches.Rectangle(
            (min_col_loc - 0.5, -0.5),
            1,
            len(heatmap.index),
            linewidth=LINEWIDTH,
            edgecolor="blue",
            facecolor="none",
        )
    )

# =====================================================
# 6. Shared colorbar
#    A single colorbar is used to indicate the magnitude
#    of L2 norm differences across all subplots.
# =====================================================
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("L2 Norm Difference", fontsize=CBAR_LABEL_FONTSIZE)
cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

# =====================================================
# 7. Figure export
#    The figure is saved with high resolution for
#    inclusion in camera-ready ACL submissions.
# =====================================================
output_path = "./heatmap_4models_grey_to_blue.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"[INFO] Figure saved to: {output_path}")
