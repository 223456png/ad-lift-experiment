"""
图表生成
========
7 张图，对应 README 里的关键结论。

数据源只有 report/metrics.json 和 report/cate_folds.json，
所以不装 duckdb、不准备 297 MB 原始数据也能重画。

运行：python src/figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import stats

OUT = "report/figures"
os.makedirs(OUT, exist_ok=True)

INK = "#1F2933"        # 正文与数字
MUTED = "#7B8794"      # 轴标签、辅助说明
GRID = "#ECEFF2"
CTRL = "#B6C0C8"       # 对照组
TRT = "#2C5F8A"        # 投放组 / 主色
LIGHT = "#9DB8CE"      # 次要柱
ACC = "#B4472F"        # 异常、亏损、需要强调的那个数
GOLD = "#B8892B"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})

M = json.load(open("report/metrics.json", encoding="utf-8"))
with open("report/cate_folds.json", encoding="utf-8") as fh:
    CF = json.load(fh)

pct0 = FuncFormatter(lambda v, _: f"{v:.0%}")
pct1 = FuncFormatter(lambda v, _: f"{v:.1%}")
pct2 = FuncFormatter(lambda v, _: f"{v:.2%}")
pct3 = FuncFormatter(lambda v, _: f"{v:.3%}")


def header(fig, title, sub=""):
    """图外的标题区：一行结论 + 一行口径说明"""
    fig.text(0.012, 0.975, title, fontsize=13.5, fontweight="bold", color=INK, va="top")
    if sub:
        fig.text(0.012, 0.888, sub, fontsize=9.5, color=MUTED, va="top")


def frame(ax, grid=True):
    """去掉上/右/左三条边，只留基线"""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D8DEE4")
    ax.tick_params(length=0, labelsize=9, pad=6)
    if grid:
        ax.yaxis.grid(True, color=GRID, lw=1.0)
        ax.set_axisbelow(True)


def save(fig, name):
    path = f"{OUT}/{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.18)
    plt.close(fig)
    print(f"  {path}")


# ══════════════════════════════════════════════════════════
# 1  依从性：把投放组按是否真的曝光拆开
# ══════════════════════════════════════════════════════════
def fig_compliance():
    c = M["compliance"]
    groups = [("对照\n未曝光", c["ctrl"]),
              ("投放\n未曝光", c["not_exposed"]),
              ("投放\n已曝光", c["exposed"])]
    colors = [CTRL, ACC, TRT]
    visits = [g[1]["visit"] for g in groups]
    convs = [g[1]["conv"] for g in groups]
    picked = visits[1] / visits[0] - 1          # 未曝光组比对照低多少
    lift = visits[2] / visits[0]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.5))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.735, bottom=0.16, wspace=0.2)
    header(fig, f"同样没看到广告，投放组的访问率比对照低 {abs(picked):.1%}",
           f"投放组按是否真的曝光拆开；已曝光的访问率是对照的 {lift:.1f} 倍，说明曝光分配本身不是随机的")

    panels = [(axes[0], visits, "访问率", pct2, pct0),
              (axes[1], convs, "转化率", pct3, pct1)]

    for ax, vals, name, fmt, ticks in panels:
        bars = ax.bar(range(3), vals, color=colors, width=0.52, zorder=3)
        ax.set_xticks(range(3))
        ax.set_xticklabels([g[0] for g in groups], fontsize=9)
        ax.set_title(name, fontsize=11, color=INK, loc="left", pad=10)
        ax.set_ylim(0, max(vals) * 1.28)
        ax.yaxis.set_major_formatter(ticks)
        frame(ax)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, fmt.func(v, None),
                    ha="center", va="bottom", fontsize=10, color=INK)

    save(fig, "01_compliance")


# ══════════════════════════════════════════════════════════
# 2  三层指标：曝光 / 访问 / 转化
# ══════════════════════════════════════════════════════════
def fig_funnel():
    f = M["funnel"]
    rows = [
        ("曝光率", f["exposures_c"], f["exposures_t"], pct2, None),
        ("访问率", f["visits_c"], f["visits_t"], pct2, (f["visits_t"] / f["n_t"]) / (f["visits_c"] / f["n_c"]) - 1),
        ("转化率", f["conv_c"], f["conv_t"], pct3, (f["conv_t"] / f["n_t"]) / (f["conv_c"] / f["n_c"]) - 1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.3))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.66, bottom=0.15, wspace=0.3)
    header(fig, "访问率 +27.1%，转化率 +59.4%，两个指标同向",
           "对照组 209.7 万人、投放组 1,188.3 万人；曝光率的分母是分配人数，不是曝光次数")

    for ax, (name, xc, xt, fmt, rel) in zip(axes, rows):
        pc = xc / f["n_c"]
        pt = xt / f["n_t"]
        bars = ax.bar([0, 1], [pc, pt], color=[CTRL, TRT], width=0.46, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["对照", "投放"], fontsize=9)
        ax.set_title(name, fontsize=11, color=INK, loc="left", pad=10)
        top = max(pc, pt)
        ax.set_ylim(0, top * 1.45 if top > 0 else 0.004)
        ax.set_xlim(-0.65, 1.65)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0 if top > 0.01 else 1))
        frame(ax)

        # 人数和增幅都放在坐标区内，避免和图外标题区打架
        ax.text(0.06, 0.94, f"{xc:,} → {xt:,} 人", transform=ax.transAxes,
                fontsize=8.6, color=MUTED, va="top")
        if rel is not None:
            ax.text(0.98, 0.94, f"{rel:+.1%}", transform=ax.transAxes, ha="right",
                    va="top", fontsize=11.5, color=TRT, fontweight="bold")

        for b, v in zip(bars, [pc, pt]):
            ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + top * 0.03,
                    "0" if v == 0 else fmt.func(v, None),
                    ha="center", va="bottom", fontsize=10, color=INK)

    save(fig, "02_funnel")


# ══════════════════════════════════════════════════════════
# 3  分层增量：一个箱子吃掉七成
# ══════════════════════════════════════════════════════════
def fig_strata():
    d = M["strata"]["detail"]
    bins = [int(r["strat_value"]) for r in d]
    inc = [(r["c_t"] - r["c_c"]) * r["n_t"] for r in d]
    total = sum(inc)
    share = [i / total for i in inc]
    top = int(np.argmax(share))

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.735, bottom=0.21)
    header(fig, f"第 {bins[top]} 箱一个箱子吃掉 {share[top]:.1%} 的增量",
           f"按 f0 五分位切；柱上是增量转化单数与占比（单位：单），五箱合计 {total:,.0f} 单")

    colors = [TRT if i == top else LIGHT for i in range(len(inc))]
    bars = ax.bar(bins, inc, color=colors, width=0.55, zorder=3)
    frame(ax)

    for b, v, s in zip(bars, inc, share):
        ax.text(b.get_x() + b.get_width() / 2, v + max(inc) * 0.10, f"{v:,.0f}",
                ha="center", va="bottom", fontsize=10.5, color=INK)
        ax.text(b.get_x() + b.get_width() / 2, v + max(inc) * 0.02, f"{s:.1%}",
                ha="center", va="bottom", fontsize=8.8, color=MUTED)

    ax.set_xticks(bins)
    ax.set_xticklabels([f"第 {b} 箱\n{r['c_c']:.3%}→{r['c_t']:.3%}"
                        for b, r in zip(bins, d)], fontsize=8.6)
    ax.set_ylim(0, max(inc) * 1.24)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    save(fig, "03_strata")


# ══════════════════════════════════════════════════════════
# 4  Qini 曲线 + 分位校准
# ══════════════════════════════════════════════════════════
def fig_qini():
    xs = np.array(CF["curve_x"])
    yv = np.array(CF["curve_y"])
    preds = np.array(CF["decile_pred"])
    incs = np.array(CF["decile_inc"])
    qini = float(CF["qini_out"])
    folds = CF["fold_qini"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.8))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.735, bottom=0.19, wspace=0.3)
    header(fig, f"样本外 Qini {qini:.2f}：前 10% 拿走 {CF['top10_cum_out']:.1%} 的增量",
           f"五折轮转交叉拟合，每个用户只被评估一次；五折 {min(folds):.3f}–{max(folds):.3f}，"
           f"极差 {max(folds)-min(folds):.3f}。右图末段分母只有 2e-5，比值不作数")

    # 左：累计增益曲线
    ax1.plot(xs, yv, color=TRT, lw=2.2, zorder=4)
    ax1.plot([0, 1], [0, 1], ls="--", color=CTRL, lw=1.2, zorder=2)
    ax1.fill_between(xs, xs, yv, color=TRT, alpha=0.07, zorder=1)
    for frac, txt, off in [(0.1, f"{CF['top10_cum_out']:.1%}", (0, 15)),
                           (0.2, f"{CF['top20_cum_out']:.1%}", (0, 15))]:
        y = float(np.interp(frac, xs, yv))
        ax1.plot([frac], [y], "o", color=ACC, ms=7, zorder=5)
        ax1.annotate(txt, xy=(frac, y), xytext=off, textcoords="offset points",
                     fontsize=10, color=ACC, fontweight="bold",
                     ha="center", va="center")
    ax1.set_title("累计增量", fontsize=11, color=INK, loc="left", pad=10)
    ax1.text(0.62, 0.50, "随机排序", fontsize=8.8, color=MUTED, rotation=34, ha="center")
    ax1.set_xlabel("按预测增量排序的样本占比", fontsize=9.2)
    ax1.set_ylabel("累计增量占比", fontsize=9.2)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)
    ax1.xaxis.set_major_formatter(pct0)
    ax1.yaxis.set_major_formatter(pct0)
    frame(ax1)

    # 右：实际/预测，一条线画在 1
    ratio = incs / preds
    xx = np.arange(1, len(ratio) + 1)
    colors = [TRT if i == 0 else (ACC if i == len(ratio) - 1 else LIGHT) for i in range(len(ratio))]
    ax2.bar(xx, ratio, color=colors, width=0.62, zorder=3)
    ax2.axhline(1, color=MUTED, lw=1.0, ls="--", zorder=4)
    ax2.text(1, ratio[0] + 0.12, f"{ratio[0]:.2f}", ha="center", va="bottom",
             fontsize=9.5, color=TRT, fontweight="bold")
    ax2.text(len(ratio), ratio[-1] * 0.72, f"{ratio[-1]:.2f}", ha="center", va="top",
             fontsize=9.5, color="white", fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_ylim(0.22, 14)
    ax2.set_yticks([0.3, 0.5, 1, 2, 4, 8])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax2.minorticks_off()
    ax2.set_xlim(0.35, 11.35)
    ax2.set_xticks(xx)
    ax2.set_xticklabels([f"{i*10+1}–{i*10+10}%" for i in range(10)],
                        fontsize=7.8, rotation=45, ha="right")
    ax2.set_title("分位校准：实际增量 ÷ 预测增量（虚线 = 1）", fontsize=11, color=INK, loc="left", pad=10)
    ax2.set_xlabel("按预测增量排序的分位", fontsize=9.2)
    frame(ax2)

    save(fig, "04_qini")


# ══════════════════════════════════════════════════════════
# 5  曝光率诊断：模型在排「会被打动」还是「会被曝光」
# ══════════════════════════════════════════════════════════
def fig_exposure():
    dec = np.array(CF["decile_exposure"])
    overall = float(CF["overall_exposure"])
    share = dec[0] / dec.sum()                  # top10 占全部曝光的份额
    cace = np.array(CF["decile_cace"])
    leveled = cace[0] / cace.sum()              # 曝光率拉平后 top10 的增量占比
    xx = np.arange(1, 11)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.6),
                                   gridspec_kw={"width_ratios": [1.75, 1]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.715, bottom=0.16, wspace=0.3)
    header(fig, f"top 10% 独占 {share:.1%} 的曝光，增量集中度里约七成来自曝光分配",
           "左图：各分位的曝光率（两臂合计，对照组曝光恒为 0）　右图：把各分位曝光率拉平后的反事实")

    colors = [TRT] + [LIGHT] * 9
    ax1.bar(xx, dec, color=colors, width=0.62, zorder=3)
    ax1.axhline(overall, color=ACC, lw=1.1, ls="--", zorder=4)
    ax1.text(10.4, overall * 1.3, f"全体 {overall:.2%}", fontsize=8.8, color=ACC, ha="right")
    ax1.text(1, dec[0] * 1.05, f"{dec[0]:.2%}（{dec[0]/overall:.1f}×）", ha="left", va="bottom",
             fontsize=9.8, color=TRT, fontweight="bold")
    ax1.set_xticks(xx)
    ax1.set_xticklabels([f"{i*10+1}–{i*10+10}%" for i in range(10)],
                        fontsize=8, rotation=45, ha="right")
    ax1.set_ylabel("曝光率", fontsize=9.2)
    ax1.yaxis.set_major_formatter(pct0)
    ax1.set_ylim(0, dec[0] * 1.55)
    ax1.set_xlim(0.3, 10.6)
    ax1.set_title("分位曝光率：从左到右跨两个数量级", fontsize=11, color=INK, loc="left", pad=10)
    frame(ax1)

    bars = ax2.bar([0, 1], [CF["top10_cum_out"], leveled], color=[TRT, LIGHT], width=0.5, zorder=3)
    for b, v in zip(bars, [CF["top10_cum_out"], leveled]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center",
                 va="bottom", fontsize=11, color=INK, fontweight="bold")
    ax2.annotate("", xy=(1, 0.92), xytext=(0, 0.92),
                 arrowprops=dict(arrowstyle="<->", color=ACC, lw=1.1))
    ax2.text(0.5, 0.935, f"−{(CF['top10_cum_out'] - leveled) * 100:.0f} 个百分点",
             ha="center", va="bottom", fontsize=9, color=ACC,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax2.set_xlim(-0.62, 1.62)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["现在的\n增量占比", "曝光率拉平后\n的增量占比"], fontsize=9)
    ax2.set_ylim(0, 1.08)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax2.yaxis.set_major_formatter(pct0)
    ax2.set_title("top 10% 的增量占比", fontsize=11, color=INK, loc="left", pad=10)
    frame(ax2)

    save(fig, "05_exposure")


# ══════════════════════════════════════════════════════════
# 6  每曝光用户成本的盈亏平衡
# ══════════════════════════════════════════════════════════
def fig_cost():
    r = M["roi"]
    cap = r["cap_per_user"]
    inc_gross = r["inc_conv"] * r["gross_per_conv"]

    costs = np.array([0.4, 0.5, 0.75, 1.0, 1.5, 1.92, 2.0, 3.0, 4.0, 6.0])
    roi = inc_gross / (r["exposed_users"] * costs)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.715, bottom=0.155)
    header(fig, f"每曝光用户成本低于 {cap:.2f} 元，这笔投放才赚钱",
           f"增量毛利 {inc_gross:,.0f} 元 ÷ {r['exposed_users']:,} 个被曝光用户；数据没有曝光次数，只能到这个粒度")

    ax.axvspan(0.3, cap, color=TRT, alpha=0.035, zorder=0)
    ax.axvspan(cap, 6.2, color=ACC, alpha=0.035, zorder=0)
    ax.plot(costs, roi, color=TRT, lw=2.2, zorder=4)
    ax.axhline(1, color=MUTED, lw=1.0, ls="--", zorder=3)
    ax.axvline(cap, color=ACC, lw=1.2, ls="--", zorder=3)

    y1 = inc_gross / (r["exposed_users"] * 1.0)
    ax.plot([1.0], [y1], "o", color=INK, ms=5, zorder=5)
    ax.text(1.0 + 0.12, y1 + 0.28, f"1 元 → ROI {y1:.2f}", fontsize=9.2, color=INK)
    y2 = inc_gross / (r["exposed_users"] * 2.0)
    ax.plot([2.0], [y2], "o", color=INK, ms=5, zorder=5)
    ax.text(2.02, 0.42, f"2 元 → ROI {y2:.2f}", fontsize=9.2, color=INK)

    ax.annotate(f"盈亏平衡 {cap:.2f} 元", xy=(cap, 1.0), xytext=(2.45, 2.35),
                fontsize=10, color=ACC, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACC, lw=1.1))

    ax.set_xlim(0.3, 6.2)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("每曝光用户成本（元 / 人）", fontsize=9.2)
    ax.set_ylabel("增量毛利 ÷ 广告成本", fontsize=9.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks([0.5, 1, 1.92, 3, 4, 6])
    ax.set_xticklabels(["0.5", "1", "1.92", "3", "4", "6"])
    frame(ax)

    save(fig, "06_cost_per_user")


# ══════════════════════════════════════════════════════════
# 7  分流比例：对照组放多少
# ══════════════════════════════════════════════════════════
def fig_design():
    p_c, rel, DAY, GROSS, C_DAY = 0.001938, 0.20, 233_000, 60.0, 5000
    lift = p_c * rel * GROSS
    za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.80)

    ps = np.linspace(0.10, 0.55, 900)
    p2 = p_c * (1 + rel)
    N = (za + zb) ** 2 * (p_c * (1 - p_c) / ps + p2 * (1 - p2) / (1 - ps)) / (p2 - p_c) ** 2
    days = N / DAY
    loss = N * ps * lift
    total = loss + days * C_DAY
    i = int(np.argmin(total))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    fig.subplots_adjust(left=0.095, right=0.90, top=0.715, bottom=0.155)
    header(fig, "对照组放到四成左右，总成本最低",
           "总成本 = 抑制损失 + 时间成本；日分配 23.3 万用户、日机会成本 5,000 元、单均毛利 60 元")

    ax.plot(ps, total, color=TRT, lw=2.4, zorder=4)
    ax.plot(ps, loss, color=CTRL, lw=1.6, ls="--", zorder=3)
    ax.plot(ps, days * C_DAY, color=GOLD, lw=1.6, ls=":", zorder=3)
    for ys, lab, col in [(total, "总成本", TRT), (days * C_DAY, "时间成本", GOLD),
                         (loss, "抑制损失", CTRL)]:
        ax.text(0.556, ys[-1], lab, color=col, fontsize=9, va="center", ha="left")

    ax.plot([ps[i]], [total[i]], "o", color=ACC, ms=8, zorder=6)
    ax.annotate(f"{ps[i]:.0%}", xy=(ps[i], total[i]), xytext=(ps[i] + 0.022, total[i] + 1600),
                fontsize=11, color=ACC, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACC, lw=1.1))

    for xv, lab in [(0.15, "原设计 15%"), (0.50, "50%")]:
        ax.axvline(xv, color="#D8DEE4", lw=1.0, zorder=1)
        ax.text(xv + 0.005, 30000, lab, fontsize=8.4, color=MUTED,
                ha="left", va="top", rotation=90)

    ax.set_xlim(0.10, 0.55)
    ax.set_ylim(0, 32000)
    ax.set_xlabel("对照组比例", fontsize=9.2)
    ax.set_ylabel("总成本（元）", fontsize=9.2)
    ax.xaxis.set_major_formatter(pct0)
    ax.set_yticks([0, 10000, 20000, 30000])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    frame(ax)

    save(fig, "07_design")


if __name__ == "__main__":
    print("生成图表 ...")
    fig_compliance()
    fig_funnel()
    fig_strata()
    fig_qini()
    fig_exposure()
    fig_cost()
    fig_design()
    print("done")
