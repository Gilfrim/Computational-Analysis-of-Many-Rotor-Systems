import os
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Data parameters
# =============================================================================
npz_g_name = "Rotor_5site_ON5_ref_weight_g_fine_30states.npz"
npz_angle_name = "Rotor_5site_ON5_ref_weight_angle_fine_30states_V1.npz"

output_file = "Figure_C0000_Weight_G_and_Gamma.pdf"


# =============================================================================
# Plot parameters
# =============================================================================
figure_size = (7.4, 3.2)
figure_dpi = 300

font_size = 10
label_size = 10
tick_size = 9

line_width = 1.35
marker_size = 4.2
marker_edge_width = 0.9

markevery_g = 3
markevery_angle = 5

show_grid = False
grid_alpha = 0.25
grid_line_width = 0.5

y_min = 0.0
y_max = 1.02

angle_x_min = -2
angle_x_max = 92
angle_x_ticks = [0, 20, 40, 60, 80]


# =============================================================================
# Okabe-Ito color palette
# =============================================================================
okabe_ito_colors = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

okabe_ito_cycle = [
    okabe_ito_colors["orange"],
    okabe_ito_colors["sky_blue"],
    okabe_ito_colors["bluish_green"],
    okabe_ito_colors["vermillion"],
    okabe_ito_colors["blue"],
    okabe_ito_colors["reddish_purple"],
    okabe_ito_colors["black"],
]

marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]


# =============================================================================
# Helper functions
# =============================================================================
def get_angle_label(angle_deg):
    angle_deg = float(angle_deg)

    if np.isclose(angle_deg, int(angle_deg)):
        return f"angle_{int(angle_deg)}"

    return f"angle_{angle_deg:g}"


def get_g_label(g_val):
    return f"g_{g_val:.2f}".replace(".", "p")


def apply_axis_style(ax):
    ax.tick_params(direction="in", top=True, right=True)

    if show_grid:
        ax.grid(True, alpha=grid_alpha, linewidth=grid_line_width)
        ax.set_axisbelow(True)


def save_figure(fig, script_dir, output_file):
    save_path = os.path.join(script_dir, output_file)

    try:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"\nsaved: {save_path}")
    except Exception as error_message:
        print(f"\nfail: {error_message}")


# =============================================================================
# Load data
# =============================================================================
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

npz_g_path = os.path.join(script_dir, npz_g_name)
npz_angle_path = os.path.join(script_dir, npz_angle_name)

if not os.path.exists(npz_g_path):
    raise FileNotFoundError(f"File not found: {npz_g_path}")

if not os.path.exists(npz_angle_path):
    raise FileNotFoundError(f"File not found: {npz_angle_path}")

data_g = np.load(npz_g_path, allow_pickle=True)
data_angle = np.load(npz_angle_path, allow_pickle=True)

angle_values = data_g["angle_values"]
g_fine = data_g["g_fine"]

g_coarse = data_angle["g_coarse"]
angles_fine = data_angle["angles_fine"]


# =============================================================================
# Plot style
# =============================================================================
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": font_size,
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# =============================================================================
# Plot
# =============================================================================
fig, (ax_g, ax_angle) = plt.subplots(
    1,
    2,
    figsize=figure_size,
    dpi=figure_dpi,
    sharey=True,
)

# -----------------------------------------------------------------------------
# Left panel: |c000...0|^2 vs g for different gamma
# -----------------------------------------------------------------------------
# Plot reference-state weight vs g.
for idx, angle_deg in enumerate(angle_values):
    angle_label = get_angle_label(angle_deg)
    ref_weight = np.asarray(data_g[f"ref_weight_{angle_label}"], dtype=float)
    ref_weight_e0 = ref_weight[:, 0]

    curve_color = okabe_ito_cycle[idx % len(okabe_ito_cycle)]
    curve_marker = marker_cycle[idx % len(marker_cycle)]

    ax_g.plot(
        g_fine,
        ref_weight_e0,
        color=curve_color,
        linestyle="-",
        linewidth=line_width,
        marker=curve_marker,
        markersize=marker_size,
        markerfacecolor="white",
        markeredgewidth=marker_edge_width,
        markevery=markevery_g,
        zorder=3,
    )

ax_g.set_xlabel(r"$g$")
ax_g.set_ylabel(r"$|c_{000\cdots0}|^2$")
ax_g.set_ylim(y_min, y_max)
apply_axis_style(ax_g)

# -----------------------------------------------------------------------------
# Right panel: |c000...0|^2 vs gamma for different g
# -----------------------------------------------------------------------------
# Plot reference-state weight vs gamma.
for idx, g_val in enumerate(g_coarse):
    g_label = get_g_label(g_val)
    ref_weight = np.asarray(data_angle[f"ref_weight_{g_label}"], dtype=float)
    ref_weight_e0 = ref_weight[:, 0]

    curve_color = okabe_ito_cycle[idx % len(okabe_ito_cycle)]
    curve_marker = marker_cycle[idx % len(marker_cycle)]

    ax_angle.plot(
        angles_fine,
        ref_weight_e0,
        color=curve_color,
        linestyle="-",
        linewidth=line_width,
        marker=curve_marker,
        markersize=marker_size,
        markerfacecolor="white",
        markeredgewidth=marker_edge_width,
        markevery=markevery_angle,
        zorder=3,
    )

ax_angle.set_xlabel(r"$\gamma$ (degrees)")
ax_angle.set_xlim(angle_x_min, angle_x_max)
ax_angle.set_xticks(angle_x_ticks)
apply_axis_style(ax_angle)

fig.tight_layout()
# Save final figure.
save_figure(fig, script_dir, output_file)

plt.show()
