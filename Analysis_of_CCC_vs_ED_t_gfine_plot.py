import os
import numpy as np
import matplotlib.pyplot as plt


npz_name = "Rotor_5site_ON5_CCC_t_amp_g_fine.npz"

fig_name_1 = "Figure_max_t1_t2_vs_g_fine.png"
pdf_name_1 = "Figure_max_t1_t2_vs_g_fine.pdf"

fig_name_2 = "Figure_norm_t2_and_scaled_ratio_t2_vs_g_fine.png"
pdf_name_2 = "Figure_norm_t2_and_scaled_ratio_t2_vs_g_fine.pdf"

angle_plot_list = np.array([0, 30, 60, 90])

show_legend = False

line_width = 1.8
marker_size = 6.0
marker_edge_width = 1.2
markevery = 3

label_fontsize = 16
tick_fontsize = 13


def get_angle_label(angle_deg):
    angle_deg = float(angle_deg)

    if np.isclose(angle_deg, int(angle_deg)):
        return f"angle_{int(angle_deg)}"

    return f"angle_{angle_deg:g}"


def plot_t_amp_g_fine_analysis():
    load_path = os.path.join(os.path.dirname(__file__), npz_name) if "__file__" in globals() else npz_name

    if not os.path.exists(load_path):
        print(f"File not found: {load_path}")
        return

    data = np.load(load_path, allow_pickle=True)

    g_fine = data["g_fine"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": tick_fontsize,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    color_list = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#D55E00",
        "#0072B2",
        "#CC79A7",
        "#000000",
    ]

    marker_list = ["o", "s", "^", "D", "v", "P", "X"]

    save_dir = os.path.dirname(__file__) if "__file__" in globals() else "."

    # ======================
    # Figure 1: max |t1| and max |t2|
    # ======================

    fig1, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.3),
        dpi=300,
        sharex=True,
    )

    for i, angle_deg in enumerate(angle_plot_list):
        label = get_angle_label(angle_deg)

        max_t1 = data[f"max_t1_{label}"]
        max_t2 = data[f"max_t2_{label}"]

        ax1.plot(
            g_fine,
            max_t1,
            lw=line_width,
            color=color_list[i],
        )

        ax1.plot(
            g_fine,
            max_t1,
            linestyle="None",
            marker=marker_list[i],
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=color_list[i],
            markeredgewidth=marker_edge_width,
            markevery=markevery,
            label=rf"$\gamma = {angle_deg:.0f}^\circ$",
        )

        ax2.plot(
            g_fine,
            max_t2,
            lw=line_width,
            color=color_list[i],
        )

        ax2.plot(
            g_fine,
            max_t2,
            linestyle="None",
            marker=marker_list[i],
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=color_list[i],
            markeredgewidth=marker_edge_width,
            markevery=markevery,
            label=rf"$\gamma = {angle_deg:.0f}^\circ$",
        )

    ax1.set_xlim(g_fine.min() - 0.02, g_fine.max() + 0.02)

    ax1.set_xlabel(r"$g$", fontsize=label_fontsize)
    ax2.set_xlabel(r"$g$", fontsize=label_fontsize)

    ax1.set_ylabel(r"max $|t_1|$", fontsize=label_fontsize)
    ax2.set_ylabel(r"max $|t_2|$", fontsize=label_fontsize)

    for ax in (ax1, ax2):
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=tick_fontsize,
            length=4.8,
            width=1.1,
        )

    if show_legend:
        ax2.legend(frameon=False, fontsize=10)

    fig1.tight_layout(w_pad=2.4)

    fig_path_1 = os.path.join(save_dir, fig_name_1)
    pdf_path_1 = os.path.join(save_dir, pdf_name_1)

    fig1.savefig(fig_path_1, dpi=300, bbox_inches="tight")
    fig1.savefig(pdf_path_1, bbox_inches="tight")

    plt.show()

    print(f"saved to {fig_path_1}")
    print(f"saved to {pdf_path_1}")

        # ======================
    # Figure 2: norm(t2) and max|t2| / ||t2||
    # ======================

    fig2, (ax3, ax4) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.3),
        dpi=300,
        sharex=True,
    )

    for i, angle_deg in enumerate(angle_plot_list):
        label = get_angle_label(angle_deg)

        norm_t2 = data[f"norm_t2_{label}"]
        ratio_t2 = data[f"ratio_t2_{label}"]

        ax3.plot(
            g_fine,
            norm_t2,
            lw=line_width,
            color=color_list[i],
        )

        ax3.plot(
            g_fine,
            norm_t2,
            linestyle="None",
            marker=marker_list[i],
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=color_list[i],
            markeredgewidth=marker_edge_width,
            markevery=markevery,
            label=rf"$\gamma = {angle_deg:.0f}^\circ$",
        )

        ax4.plot(
            g_fine,
            ratio_t2,
            lw=line_width,
            color=color_list[i],
        )

        ax4.plot(
            g_fine,
            ratio_t2,
            linestyle="None",
            marker=marker_list[i],
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=color_list[i],
            markeredgewidth=marker_edge_width,
            markevery=markevery,
            label=rf"$\gamma = {angle_deg:.0f}^\circ$",
        )

    ax3.set_xlim(g_fine.min() - 0.02, g_fine.max() + 0.02)

    ax3.set_xlabel(r"$g$", fontsize=label_fontsize)
    ax4.set_xlabel(r"$g$", fontsize=label_fontsize)

    ax3.set_ylabel(r"$\|t_2\|$", fontsize=label_fontsize)
    ax4.set_ylabel(r"$\max|t_2|/\|t_2\|$", fontsize=label_fontsize)

    for ax in (ax3, ax4):
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=tick_fontsize,
            length=4.8,
            width=1.1,
        )

    if show_legend:
        ax4.legend(frameon=False, fontsize=10)

    fig2.tight_layout(w_pad=2.4)

    fig_name_2 = "Figure_norm_t2_and_ratio_t2_vs_g_fine.png"
    pdf_name_2 = "Figure_norm_t2_and_ratio_t2_vs_g_fine.pdf"

    fig_path_2 = os.path.join(save_dir, fig_name_2)
    pdf_path_2 = os.path.join(save_dir, pdf_name_2)

    fig2.savefig(fig_path_2, dpi=300, bbox_inches="tight")
    fig2.savefig(pdf_path_2, bbox_inches="tight")

    plt.show()

    print(f"saved to {fig_path_2}")
    print(f"saved to {pdf_path_2}")


plot_t_amp_g_fine_analysis()