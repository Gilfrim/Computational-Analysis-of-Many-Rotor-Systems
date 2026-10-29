import os
import glob
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Data parameters
# =============================================================================
file_name = "Heisenberg_12site_CCC_energy_basis_30states.npz"

plot_labels = ["4x3", "3x4", "2x6"]
include_6x2 = False

if include_6x2:
    plot_labels = ["6x2", "4x3", "3x4", "2x6"]

plot_per_site = True

# "absolute" gives |E_CCC - E_select| / |E_select| * 100
# "signed" gives (E_CCC - E_select) / |E_select| * 100
error_mode = "signed"

output_file = "Figure_Magnetic_CCC_vs_Eselect.pdf"


# =============================================================================
# Plot parameters
# =============================================================================
figure_size = (7.4, 3.4)
figure_dpi = 300

font_size = 10
label_size = 10
tick_size = 9

line_width = 1.4
marker_size = 5.0
marker_edge_width = 1.0

reference_linestyle = "-"
ccc_linestyle = "None"
error_linestyle = "-"

zero_line_width = 0.8

show_grid = False
grid_alpha = 0.20
grid_line_width = 0.5

use_manual_g_range = True
g_min = 0.5
g_max = 2.0

use_fixed_error_ylim = False
fixed_error_ylim = (0.0, 20.0)

error_y_padding_fraction = 0.08


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
    "gray": "#7F7F7F",
}

okabe_ito_cycle = [
    okabe_ito_colors["orange"],
    okabe_ito_colors["sky_blue"],
    okabe_ito_colors["bluish_green"],
    okabe_ito_colors["vermillion"],
    okabe_ito_colors["blue"],
    okabe_ito_colors["reddish_purple"],
]

marker_cycle = ["o", "s", "^", "D", "v", "P"]


# =============================================================================
# Helper functions
# =============================================================================
def find_npz_file(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

    direct_path = os.path.join(script_dir, file_name)

    if os.path.exists(direct_path):
        return direct_path

    candidates = sorted(
        glob.glob(os.path.join(script_dir, "Heisenberg_12site_CCC_energy_basis*.npz")),
        key=os.path.getmtime,
        reverse=True,
    )

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Cannot find {file_name}, and no matching Heisenberg_12site_CCC_energy_basis*.npz files were found."
        )

    print(f"Using newest matching file:\n{candidates[0]}")
    return candidates[0]


def calculate_error_percent(reference_values, test_values, mode="absolute"):
    reference_values = np.asarray(reference_values, dtype=float)
    test_values = np.asarray(test_values, dtype=float)

    error = np.full_like(test_values, np.nan, dtype=float)

    valid = (
        np.isfinite(reference_values)
        & np.isfinite(test_values)
        & (np.abs(reference_values) > 1.0e-14)
    )

    if mode == "absolute":
        error[valid] = (
            np.abs(test_values[valid] - reference_values[valid])
            / np.abs(reference_values[valid])
            * 100.0
        )
    elif mode == "signed":
        error[valid] = (
            (test_values[valid] - reference_values[valid])
            / np.abs(reference_values[valid])
            * 100.0
        )
    else:
        raise ValueError("error_mode must be either 'absolute' or 'signed'.")

    return error


def apply_axis_style(ax):
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=tick_size,
        length=4.2,
        width=1.0,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    if show_grid:
        ax.grid(True, alpha=grid_alpha, linewidth=grid_line_width)
        ax.set_axisbelow(True)


def add_zero_line(ax):
    ax.axhline(
        0.0,
        color=okabe_ito_colors["black"],
        linewidth=zero_line_width,
        linestyle="-",
        zorder=1,
    )


def apply_error_axis_limit(ax, error_collection):
    if use_fixed_error_ylim:
        ax.set_ylim(*fixed_error_ylim)
        return

    error_collection = np.asarray(error_collection, dtype=float)
    finite_values = error_collection[np.isfinite(error_collection)]

    if finite_values.size == 0:
        ax.set_ylim(-1.0, 1.0)
        return

    error_min = np.min(finite_values)
    error_max = np.max(finite_values)

    if error_mode == "absolute":
        error_min = min(0.0, error_min)

    if abs(error_max - error_min) < 1.0e-12:
        center = 0.5 * (error_min + error_max)
        ax.set_ylim(center - 1.0, center + 1.0)
        return

    error_range = error_max - error_min
    error_padding = error_y_padding_fraction * error_range

    ax.set_ylim(error_min - error_padding, error_max + error_padding)


def label_to_latex(label):
    left, right = label.split("x")
    return rf"${left}\times{right}$"


def require_key(data, key):
    if key not in data.files:
        raise KeyError(
            f"Missing key: {key}\n"
            f"Available keys:\n{list(data.files)}"
        )


def save_figure(fig, script_dir, output_file):
    save_path = os.path.join(script_dir, output_file)

    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"\nsaved: {save_path}")


# =============================================================================
# Load data
# =============================================================================
# Load magnetic CCC/E-select comparison.
load_path = find_npz_file(file_name)
script_dir = os.path.dirname(load_path)

data = np.load(load_path, allow_pickle=True)

require_key(data, "g_list")
require_key(data, "site")

g_list = np.asarray(data["g_list"], dtype=float)
site = int(data["site"])

g_mask = np.ones_like(g_list, dtype=bool)

if use_manual_g_range:
    g_mask &= g_list >= g_min
    g_mask &= g_list <= g_max

g_plot = g_list[g_mask]


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
fig, (ax_energy, ax_error) = plt.subplots(
    1,
    2,
    figsize=figure_size,
    dpi=figure_dpi,
)

error_collection = []

# Plot selected combined bases.
for idx, label in enumerate(plot_labels):
    e_select_key = f"E_select_ED_{label}"
    e_ccc_key = f"E_CCC_energy_{label}"
    residual_key = f"final_residual_{label}"
    converged_key = f"converged_{label}"

    require_key(data, e_select_key)
    require_key(data, e_ccc_key)

    E_select = np.asarray(data[e_select_key], dtype=float)[g_mask]
    E_CCC = np.asarray(data[e_ccc_key], dtype=float)[g_mask]

    # Compare CCC against selected ED energy.
    error_percent = calculate_error_percent(
        reference_values=E_select,
        test_values=E_CCC,
        mode=error_mode,
    )

    error_collection.append(error_percent)

    if plot_per_site:
        E_select_plot = E_select / site
        E_CCC_plot = E_CCC / site
    else:
        E_select_plot = E_select
        E_CCC_plot = E_CCC

    color = okabe_ito_cycle[idx % len(okabe_ito_cycle)]
    marker = marker_cycle[idx % len(marker_cycle)]

    # -------------------------------------------------------------------------
    # Left panel: E_select and E_CCC
    # -------------------------------------------------------------------------
    # ED selected-state reference.
    ax_energy.plot(
        g_plot,
        E_select_plot,
        color=color,
        linestyle=reference_linestyle,
        linewidth=line_width,
        zorder=2,
    )

    # CCC result.
    ax_energy.plot(
        g_plot,
        E_CCC_plot,
        color=color,
        linestyle=ccc_linestyle,
        marker=marker,
        markersize=marker_size,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=marker_edge_width,
        zorder=3,
    )

    # -------------------------------------------------------------------------
    # Right panel: error vs E_select
    # -------------------------------------------------------------------------
    # Signed or absolute percent error.
    ax_error.plot(
        g_plot,
        error_percent,
        color=color,
        linestyle=error_linestyle,
        linewidth=line_width,
        marker=marker,
        markersize=marker_size,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=marker_edge_width,
        zorder=3,
    )

    if residual_key in data.files:
        residual_values = np.asarray(data[residual_key], dtype=float)[g_mask]
        print(
            f"{label}: residual min={np.nanmin(residual_values):.3e}, "
            f"max={np.nanmax(residual_values):.3e}"
        )

    if converged_key in data.files:
        converged_values = np.asarray(data[converged_key], dtype=bool)[g_mask]
        print(f"{label}: converged {np.sum(converged_values)} / {len(converged_values)}")

# =============================================================================
# Axes
# =============================================================================
ax_energy.set_xlabel(r"$g$")

if plot_per_site:
    ax_energy.set_ylabel(r"$E/N_{\mathrm{site}}$")
else:
    ax_energy.set_ylabel(r"$E$")

if error_mode == "absolute":
    ax_error.set_ylabel(r"Error vs $E_{\mathrm{select}}$ (\%)")
else:
    ax_error.set_ylabel(r"Signed error vs $E_{\mathrm{select}}$ (\%)")
    add_zero_line(ax_error)

ax_error.set_xlabel(r"$g$")

apply_axis_style(ax_energy)
apply_axis_style(ax_error)

apply_error_axis_limit(ax_error, np.asarray(error_collection))

fig.tight_layout()
# Save final figure.
save_figure(fig, script_dir, output_file)

plt.show()


# =============================================================================
# Diagnostics
# =============================================================================
# Print run summary.
print("\nData file used:")
print(load_path)

print("\nLabels plotted:")
print(plot_labels)

print("\nError definition:")
if error_mode == "absolute":
    print("|E_CCC - E_select| / |E_select| * 100%")
else:
    print("(E_CCC - E_select) / |E_select| * 100%")

print("\nCaption helper:")
caption_text = (
    "Comparison of CCC and ED selected-state energies for the 12-site Heisenberg model. "
    "Left: energy per site as a function of interaction strength $g$; solid curves denote "
    "$E_{\\mathrm{select}}$ from ED and open markers denote $E_{\\mathrm{CCC}}$. "
    "Right: percent error of $E_{\\mathrm{CCC}}$ relative to $E_{\\mathrm{select}}$. "
    "Different colors and markers indicate different combined-site bases: "
    + ", ".join(label_to_latex(label) for label in plot_labels)
    + "."
)

print(caption_text)
