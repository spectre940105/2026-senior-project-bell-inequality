import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path


# 若系統有微軟正黑體，中文標題就能正常顯示。
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


OUTPUT_DIR = Path(__file__).resolve().parent / "figures"


def draw_bell_plot(w, bell_value, wc, title, formula, output_stem):
    """繪製 Bell inequality 最大值隨 Werner-state 參數 w 的變化。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)

    # Bell inequality 的連續理論曲線
    ax.plot(w, bell_value, linewidth=2.4, color="#2563eb", label=formula)

    # 每隔 0.05 顯示一個實際數據點，讓圖不只是單純的理論直線。
    data_index = np.arange(0, len(w), 50)
    ax.scatter(
        w[data_index],
        bell_value[data_index],
        s=24,
        facecolor="white",
        edgecolor="#2563eb",
        linewidth=1.2,
        zorder=4,
        label="Numerical data (step = 0.05)",
    )

    # y = 0 是違背 Bell inequality 與否的分界線
    ax.axhline(0, linewidth=1.5, color="#202020", linestyle="--", label="Classical bound: I = 0")

    # 臨界 w 值
    ax.axvline(wc, linewidth=1.6, color="#dc2626", linestyle=":", label=fr"Critical $w_c={wc:.4f}$")
    ax.scatter([wc], [0], s=46, color="#dc2626", zorder=5)

    # 標示沒有白噪音時（w = 1）的最大值。
    ax.scatter([1], [bell_value[-1]], s=46, color="#7c3aed", zorder=5)
    ax.annotate(
        fr"$I(1)={bell_value[-1]:.4f}$",
        xy=(1, bell_value[-1]),
        xytext=(-72, -20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#7c3aed"},
        fontsize=10,
    )

    # 將違背區域淡淡標出：I > 0
    ax.fill_between(w, 0, bell_value, where=(bell_value > 0), color="#22c55e", alpha=0.18,
                    label="Violation region: I > 0")

    ax.annotate(
        fr"$w_c={wc:.4f}$" + "\n" + fr"noise $R_c=1-w_c={1-wc:.4f}$",
        xy=(wc, 0),
        xytext=(wc - 0.28, bell_value.max() * 0.55),
        arrowprops={"arrowstyle": "->", "color": "#dc2626"},
        fontsize=10,
    )

    ax.set_title(title, fontsize=15)
    ax.set_xlabel(r"Entangled-state fraction $w$  (white-noise fraction $R=1-w$)")
    ax.set_ylabel("Maximum Bell value")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        fontsize=9,
        loc="upper left",
    )
    fig.tight_layout()
    # PDF 是向量圖，適合直接放入 LaTeX；PNG 方便一般預覽。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{output_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{output_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    # 對 w=0 到 w=1 取 1001 個點；這就是數值掃描的部分。
    w = np.linspace(0.0, 1.0, 1001)

    # 論文中的式 (4.5)：I_2222^max(w) = -1/2 + (sqrt(2)/2) w
    i2222 = -0.5 + (np.sqrt(2) / 2) * w
    wc2222 = 1 / np.sqrt(2)

    # 論文中的式 (4.8)：I_3322^max(w) = -1 + (5/4) w
    i3322 = -1.0 + 1.25 * w
    wc3322 = 0.8

    draw_bell_plot(
        w,
        i2222,
        wc2222,
        r"Werner white noise effect on $I_{2222}$",
        r"$I_{2222}^{max}(w)=-\frac{1}{2}+\frac{\sqrt{2}}{2}w$",
        "I2222_vs_w",
    )

    draw_bell_plot(
        w,
        i3322,
        wc3322,
        r"Werner white noise effect on $I_{3322}$",
        r"$I_{3322}^{max}(w)=-1+\frac{5}{4}w$",
        "I3322_vs_w",
    )

if __name__ == "__main__":
    main()
