"""
图表生成
========
6 张图，对应 README 里的关键结论。

风格上刻意做减法：不用 suptitle、不加长图注、去掉网格，
关键信息直接标在图上。

运行：python src/figures.py
"""
import json
import os

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy import optimize

OUT = "report/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#BBBBBB",
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "font.size": 10.5,
    "text.color": "#222222",
    "axes.labelcolor": "#444444",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
})

BLUE = "#2C6E9B"
RED = "#B03A2E"
GREY = "#9AA5AD"
DARK = "#333333"

M = json.load(open("report/metrics.json", encoding="utf-8"))
FEATS = [f"f{i}" for i in range(12)]

pct0 = FuncFormatter(lambda v, _: f"{v:.0%}")
pct1 = FuncFormatter(lambda v, _: f"{v:.1%}")
pct2 = FuncFormatter(lambda v, _: f"{v:.2%}")


def save(fig, name):
    path = f"{OUT}/{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)
    print(f"  {path}")


def clean(ax):
    """只留水平参考线，不要全网格"""
    ax.yaxis.grid(True, color="#E9E9E9", lw=0.8)
    ax.set_axisbelow(True)


# ══════════════════════════════════════════════════════════
# 1  依从性：三组对比
# ══════════════════════════════════════════════════════════
def fig_compliance():
    c = M["compliance"]
    labels = ["对照\n(全未曝光)", "投放\n已曝光", "投放\n未曝光"]
    visits = [c["ctrl"]["visit"], c["exposed"]["visit"], c["not_exposed"]["visit"]]
    convs = [c["ctrl"]["conv"], c["exposed"]["conv"], c["not_exposed"]["conv"]]
    colors = [GREY, BLUE, RED]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    fig.subplots_adjust(wspace=0.3, bottom=0.2)

    for ax, vals, title, fmt, fm in [
        (axes[0], visits, "访问率（未曝光组比对照低 8.7%）", "{:.2%}", pct0),
        (axes[1], convs, "转化率", "{:.3%}", pct1),
    ]:
        bars = ax.bar(range(3), vals, color=colors, width=0.55, zorder=3)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_title(title, fontsize=11, color=DARK, pad=12, loc="left")
        ax.set_ylim(0, max(vals) * 1.18)
        ax.yaxis.set_major_formatter(fm)
        clean(ax)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.015, fmt.format(v),
                    ha="center", va="bottom", fontsize=10.5, color=DARK)

    for ax, vals in [(axes[0], visits), (axes[1], convs)]:
        ax.axhline(vals[0], color=RED, ls="--", lw=1.1, alpha=0.75, zorder=2)



    save(fig, "01_compliance")


# ══════════════════════════════════════════════════════════
# 2  三层指标
# ══════════════════════════════════════════════════════════
def fig_funnel():
    f = M["funnel"]
    rows = [
        ("曝光率", f["exposures_c"] / f["n_c"], f["exposures_t"] / f["n_t"], pct2),
        ("访问率", f["visits_c"] / f["n_c"], f["visits_t"] / f["n_t"], pct2),
        ("转化率", f["conv_c"] / f["n_c"], f["conv_t"] / f["n_t"], pct1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.0))
    fig.subplots_adjust(wspace=0.42, bottom=0.18)

    for ax, (name, pc, pt, fm) in zip(axes, rows):
        bars = ax.bar([0, 1], [pc, pt], color=[GREY, BLUE], width=0.5, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["对照", "投放"], fontsize=9.5)
        ax.set_title(name, fontsize=11.5, color=DARK, pad=10, loc="left")
        top = max(pc, pt)
        ax.set_ylim(0, top * 1.45 if top > 0 else 0.01)
        ax.yaxis.set_major_formatter(fm)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        clean(ax)
        for b, v in zip(bars, [pc, pt]):
            t = "0" if v == 0 else (f"{v:.5%}" if v < 0.01 else f"{v:.3%}")
            ax.text(b.get_x() + b.get_width() / 2, max(v, 0) * 1.02 + top * 0.01,
                    t, ha="center", va="bottom", fontsize=10, color=DARK)
        if pc > 0:
            ax.text(0.5, top * 1.28, f"{pt/pc-1:+.1%}", ha="center",
                    fontsize=11, color=BLUE, fontweight="bold")
        else:
            ax.text(0.5, top * 1.28, "对照组恒为 0", ha="center",
                    fontsize=8.5, color=GREY)

    save(fig, "02_funnel")


# ══════════════════════════════════════════════════════════
# 3  分层增量：只留柱状图，累计曲线信息重复
# ══════════════════════════════════════════════════════════
def fig_strata():
    d = M["strata"]["detail"]
    bins = [int(r["strat_value"]) for r in d]
    inc = [(r["c_t"] - r["c_c"]) * r["n_t"] for r in d]
    total = sum(inc)
    share = [i / total for i in inc]
    top = int(np.argmax(share))

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    fig.subplots_adjust(bottom=0.18)
    colors = [BLUE if i == top else "#B8CBD9" for i in range(len(inc))]
    bars = ax.bar(bins, inc, color=colors, width=0.55, zorder=3)
    clean(ax)

    for b, v, s in zip(bars, inc, share):
        ax.text(b.get_x() + b.get_width() / 2, v + max(inc) * 0.02,
                f"{s:.1%}", ha="center", va="bottom",
                fontsize=10.5, color=DARK if s > 0.5 else "#777777",
                fontweight="bold" if s > 0.5 else "normal")

    ax.set_xticks(bins)
    ax.set_xticklabels([f"第 {b} 箱" for b in bins], fontsize=9.5)
    ax.set_ylabel("增量转化（单）", fontsize=9.5)
    ax.set_ylim(0, max(inc) * 1.2)
    ax.set_title(f"增量高度集中在第 {bins[top]} 箱：占全部增量的 {share[top]:.0%}",
                 fontsize=11.5, color=DARK, loc="left", pad=12)
    ax.text(0.99, 0.92, f"合计 {total:,.0f} 单", transform=ax.transAxes,
            ha="right", fontsize=9, color="#777777")

    save(fig, "03_strata")


# ══════════════════════════════════════════════════════════
# 4  Qini + 分位校准
# ══════════════════════════════════════════════════════════
def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def _fit_logit(X, y, l2=1e-6, maxiter=300):
    Xd = np.column_stack([np.ones(len(X)), X])

    def nll(b):
        z = Xd @ b
        return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(b[1:] ** 2)

    return optimize.minimize(nll, np.zeros(Xd.shape[1]), method="L-BFGS-B",
                             options={"maxiter": maxiter}).x


def fig_qini():
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    d = con.execute("""
        SELECT f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,treatment,conversion
        FROM read_csv('data/criteo-uplift-v2.1.csv.gz', header=true)
        WHERE hash(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,treatment,conversion,visit,exposure) % 10000 < 430
    """).df()
    X = d[FEATS].values.astype(float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = d.conversion.values.astype(float)
    w = d.treatment.values.astype(int)

    b1 = _fit_logit(Xs[w == 1], y[w == 1])
    b0 = _fit_logit(Xs[w == 0], y[w == 0])
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    cate = _sigmoid(Xd @ b1) - _sigmoid(Xd @ b0)

    order = np.argsort(-cate)
    yy, ww = y[order], w[order].astype(bool)
    n = len(yy)
    incs, sizes, preds = [], [], []
    for b in np.array_split(np.arange(n), 10):
        yb, wb = yy[b], ww[b]
        if wb.sum() == 0 or (~wb).sum() == 0:
            continue
        incs.append(yb[wb].mean() - yb[~wb].mean())
        sizes.append(len(b))
        preds.append(cate[order][b].mean())
    incs, sizes, preds = np.array(incs), np.array(sizes), np.array(preds)

    cum = np.cumsum(incs * sizes)
    total = cum[-1] if abs(cum[-1]) > 1e-12 else 1e-12
    xs = np.concatenate([[0.0], np.cumsum(sizes) / n])
    yv = np.concatenate([[0.0], cum / total])
    qini = 2 * float(np.trapezoid(yv, xs)) - 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.5))
    fig.subplots_adjust(wspace=0.32, bottom=0.2)

    ax1.plot(xs, yv, "-", color=BLUE, lw=2.2, zorder=3)
    ax1.plot([0, 1], [0, 1], "--", color=GREY, lw=1.1)
    ax1.fill_between(xs, xs, yv, color=BLUE, alpha=0.08)
    for frac in (0.1, 0.2):
        k = max(1, int(round(frac * len(incs))))
        ax1.plot([frac], [yv[k]], "o", color=RED, ms=7, zorder=5)
        ax1.text(frac + 0.03, yv[k] + 0.05, f"{yv[k]:.0%}",
                 fontsize=9.5, color=RED)
    ax1.set_xlabel("按预测增量排序的样本占比", fontsize=9.5)
    ax1.set_ylabel("累计增量占比", fontsize=9.5)
    ax1.set_title(f"排序有效（Qini {qini:.2f}）", fontsize=11.5,
                  color=DARK, loc="left", pad=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.03)
    ax1.xaxis.set_major_formatter(pct0)
    ax1.yaxis.set_major_formatter(pct0)
    clean(ax1)

    # 用「预测 vs 实际」并排，比比值法稳（末尾分位预测值趋近 0，比值会爆炸）
    k = len(incs)
    xx = np.arange(k)
    wbar = 0.38
    ax2.bar(xx - wbar / 2, preds, wbar, color="#B8CBD9", zorder=3, label="模型预测")
    ax2.bar(xx + wbar / 2, incs, wbar, color=BLUE, zorder=3, label="实际观测")
    ax2.axhline(0, color="#888888", lw=0.9, zorder=2)
    ax2.set_xticks(xx)
    ax2.set_xticklabels([f"{int((i+1)*10)}%" for i in range(k)], fontsize=8.5)
    ax2.set_xlabel("按预测增量排序的分位", fontsize=9.5)
    ax2.set_ylabel("增量转化率", fontsize=9.5)
    ax2.set_title("但只有前 20% 校准得住", fontsize=11.5,
                  color=DARK, loc="left", pad=10)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right")
    clean(ax2)
    lo = min(incs.min(), preds.min())
    hi = max(incs.max(), preds.max())
    ax2.set_ylim(lo * 1.35 if lo < 0 else 0, hi * 1.35)
    # 标出前两段校准好的区间
    ax2.axvspan(-0.5, 1.5, color=BLUE, alpha=0.05, zorder=1)
    ax2.text(0.5, hi * 1.08, "量级对得上", ha="center", fontsize=8.5, color=BLUE)

    save(fig, "04_qini")


# ══════════════════════════════════════════════════════════
# 5  CPM 敏感性
# ══════════════════════════════════════════════════════════
def fig_cpm():
    r = M["roi"]
    exposed = r["exposed_users"]         # 被曝光用户数（user 级，不是曝光次数）
    inc_gross = r["inc_gross"]
    cap = r["cap_per_user"]              # 盈亏平衡的每曝光用户成本上限

    costs = np.array([0.25, 0.5, 1.0, 1.92, 3.0, 4.0, 6.0, 10.0])
    roi = inc_gross / (exposed * costs)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    fig.subplots_adjust(bottom=0.2)
    ax.plot(costs, roi, "-", color=BLUE, lw=2.2, zorder=3)
    ax.plot(costs, roi, "o", color=BLUE, ms=4, zorder=4)
    ax.axhline(1, color=RED, ls="--", lw=1.2, zorder=2)
    ax.axvline(cap, color=RED, ls=":", lw=1.2, zorder=2)
    ax.fill_between(costs, 0.3, roi, where=(costs <= cap), color=BLUE, alpha=0.06)
    ax.fill_between(costs, 0.3, roi, where=(costs > cap), color=RED, alpha=0.08)

    ax.annotate(f"盈亏平衡\n{cap:,.2f} 元/人",
                xy=(cap, 1.0), xytext=(cap * 0.30, 2.6),
                fontsize=10, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("每个被曝光用户的成本（元 / 人）", fontsize=9.5)
    ax.set_ylabel("增量毛利 / 总成本", fontsize=9.5)
    ax.set_title("每曝光用户成本低于 1.92 元，投放不亏", fontsize=11.5,
                 color=DARK, loc="left", pad=12)
    ax.set_ylim(0.3, roi[0] * 1.15)
    clean(ax)

    save(fig, "05_cpm")


# ══════════════════════════════════════════════════════════
# 6  分流比例权衡
# ══════════════════════════════════════════════════════════
def fig_design():
    from scipy import stats
    p_c, rel, DAY, GROSS, C_DAY = 0.001938, 0.20, 233_000, 60.0, 5000
    lift = p_c * rel * GROSS
    za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.80)

    ps = np.linspace(0.08, 0.55, 80)
    p2 = p_c * (1 + rel)
    N = (za + zb) ** 2 * (p_c * (1 - p_c) / ps + p2 * (1 - p2) / (1 - ps)) / (p2 - p_c) ** 2
    days = N / DAY
    loss = N * ps * lift
    total = loss + days * C_DAY

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    fig.subplots_adjust(bottom=0.2)
    ax.plot(ps, total, lw=2.2, color=BLUE, zorder=3, label="总成本")
    ax.plot(ps, loss, lw=1.4, color=GREY, ls="--", zorder=2, label="抑制损失")
    ax.plot(ps, days * C_DAY, lw=1.4, color="#C9A227", ls=":", zorder=2,
            label="时间成本")

    i = int(np.argmin(total))
    ax.plot([ps[i]], [total[i]], "o", color=RED, ms=9, zorder=5)
    ax.annotate(f"{ps[i]:.0%}", xy=(ps[i], total[i]),
                xytext=(ps[i] + 0.045, total[i] * 1.30),
                fontsize=11, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))

    for xv, lab in [(0.15, "原设计 15%"), (0.5, "50%")]:
        ax.axvline(xv, color="#CCCCCC", ls="-.", lw=1, zorder=1)
        ax.text(xv, total.max() * 0.97, lab, fontsize=8.5, color="#888888",
                rotation=90, va="top", ha="right")

    ax.set_xlabel("对照组比例", fontsize=9.5)
    ax.set_ylabel("总成本（元）", fontsize=9.5)
    ax.set_title("对照组 40% 时总成本最低，但 30–50% 差别很小", fontsize=11.5,
                 color=DARK, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_xlim(0.08, 0.55)
    ax.xaxis.set_major_formatter(pct0)
    clean(ax)

    save(fig, "06_design")


if __name__ == "__main__":
    print("生成图表 ...")
    fig_compliance()
    fig_funnel()
    fig_strata()
    fig_qini()
    fig_cpm()
    fig_design()
    print("done")
