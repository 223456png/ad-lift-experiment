"""
Criteo-UPLIFT v2.1 —— 广告增量性 A/B 实验 · 分析层
====================================================
对应 JD：实验方案设计 → 分层观测 → 链路指标 → SQL/Python 归因 → 收益验证 → 稳定性评估

运行：
    python src/analysis.py                 # 全量
    python src/analysis.py --aov 200 --margin 0.3

输出：终端报告 + report/metrics.json
"""
import argparse
import json
import os
import sys
import time

import duckdb

from dbutil import connect as duck_connect
import numpy as np
import pandas as pd
from scipy import stats, optimize

DATA = "data/criteo-uplift-v2.1.csv.gz"
FEATS = [f"f{i}" for i in range(12)]
R = {}          # 结果收集器，最后落 JSON


# ══════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logit(X, y, l2=1e-6, maxiter=400):
    """L2 正则的 logistic 回归（L-BFGS + 解析梯度），避免依赖 statsmodels。

    为什么必须给解析梯度：数值差分每迭代要 14 次函数评估（13 参数 + 1），
    在 280 万行训练集上单次拟合要几分钟；解析梯度把单次拟合降到秒级，
    五折交叉拟合才跑得动。
    """
    Xd = np.column_stack([np.ones(len(X)), X])

    def nll(b):
        z = Xd @ b
        return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(b[1:] ** 2)

    def grad(b):
        z = Xd @ b
        g = Xd.T @ (sigmoid(z) - y)
        g[1:] += 2 * l2 * b[1:]
        return g

    res = optimize.minimize(nll, np.zeros(Xd.shape[1]), jac=grad, method="L-BFGS-B",
                            options={"maxiter": maxiter})
    return res.x


def two_prop(x1, n1, x2, n2, alpha=0.05):
    """两独立比例检验（对照组 1，实验组 2）"""
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se_pool = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se_pool if se_pool > 0 else 0.0
    pval = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    half = stats.norm.ppf(1 - alpha / 2) * se
    return dict(p_c=p1, p_t=p2, diff=p2 - p1, rel=(p2 - p1) / p1,
                lo=p2 - p1 - half, hi=p2 - p1 + half,
                rel_lo=(p2 - p1 - half) / p1, rel_hi=(p2 - p1 + half) / p1,
                se=se, z=z, p=pval)


def fmt_p(p):
    return "<1e-16" if p < 1e-16 else f"{p:.3g}"


def banner(t):
    print(f"\n{'='*74}\n{t}\n{'='*74}")


# ══════════════════════════════════════════════════════════════
# 1. 载入 + SRM
# ══════════════════════════════════════════════════════════════
def stage_srm(con):
    banner("[1] SRM 分流健康度（RCT 分配的完整性）")
    df = con.execute("""
        SELECT treatment, COUNT(*) n FROM raw GROUP BY treatment ORDER BY treatment
    """).df()
    n_c, n_t = int(df.loc[df.treatment == 0, "n"].iloc[0]), int(df.loc[df.treatment == 1, "n"].iloc[0])
    total = n_c + n_t
    obs = np.array([n_c, n_t], float)
    # 设计比例 85:15（数据集的原始分流设计）
    exp = np.array([0.15, 0.85]) * total
    chi2, p = stats.chisquare(obs, exp)

    print(f"  control   n={n_c:>12,}  share={n_c/total:7.4%}  (设计 15.00%)")
    print(f"  treatment n={n_t:>12,}  share={n_t/total:7.4%}  (设计 85.00%)")
    print(f"  卡方={chi2:.4f}  p={fmt_p(p)}  -> {'通过' if p > 0.001 else '不通过'}")
    print(f"  组间样本比 1:{n_t/n_c:.2f} —— 不均衡会放大方差，"
          f"有效样本量 n_eff = {(1/n_c + 1/n_t)**-1:,.0f}")

    R["srm"] = dict(n_c=n_c, n_t=n_t, share_t=n_t / total, chi2=float(chi2), p=float(p),
                    n_eff=float((1 / n_c + 1 / n_t) ** -1))
    return n_c, n_t


# ══════════════════════════════════════════════════════════════
# 2. 协变量平衡检查（A/B balance）
# ══════════════════════════════════════════════════════════════
def stage_aa(con):
    banner("[2] 协变量平衡检查（A/B balance，严格说不叫 AA 检验）")
    print("  说明：AA 检验的定义是两个**对照组之间**的对比，这里比的是")
    print("        treated vs control 的特征分布，本质是随机化质量检查。")
    print("        14M 样本下 SMD 几乎必然统计显著，看的是实务阈值（<0.1 可忽略）。")
    aggs = []
    for f in FEATS:
        aggs += [f"AVG({f}) FILTER (WHERE treatment=1) AS {f}_m1",
                 f"AVG({f}) FILTER (WHERE treatment=0) AS {f}_m0",
                 f"VAR_SAMP({f}) FILTER (WHERE treatment=1) AS {f}_v1",
                 f"VAR_SAMP({f}) FILTER (WHERE treatment=0) AS {f}_v0"]
    r = con.execute(f"SELECT {', '.join(aggs)} FROM raw").df().iloc[0]

    rows = []
    for f in FEATS:
        smd = abs(r[f"{f}_m1"] - r[f"{f}_m0"]) / np.sqrt((r[f"{f}_v1"] + r[f"{f}_v0"]) / 2)
        rows.append((f, r[f"{f}_m1"], r[f"{f}_m0"], smd))
    tab = pd.DataFrame(rows, columns=["feature", "mean_t", "mean_c", "smd"])
    bad = tab[tab.smd > 0.1]
    print(tab.to_string(index=False, float_format=lambda x: f"{x: .5f}"))
    print(f"\n  最大标准化均值差 SMD = {tab.smd.max():.5f}  "
          f"(>0.1 为不平衡) -> {'通过' if len(bad)==0 else '存在不平衡: '+str(bad.feature.tolist())}")
    R["aa"] = dict(max_smd=float(tab.smd.max()), n_unbalanced=int(len(bad)))
    return tab


# ══════════════════════════════════════════════════════════════
# 3. 链路漏斗 + ITT 主效应
# ══════════════════════════════════════════════════════════════
def stage_itt(con):
    banner("[3] 链路漏斗 + ITT 主效应")
    f = con.execute("""
        SELECT treatment, COUNT(*) n, SUM(exposure) exp_, SUM(visit) v, SUM(conversion) c
        FROM raw GROUP BY treatment ORDER BY treatment
    """).df().set_index("treatment")
    c_, t_ = f.loc[0], f.loc[1]

    print("  链路：分配 → 曝光 → 访问 → 转化")
    print(f"  {'arm':<12}{'l1_assigned':>13}{'l2_exposed':>13}{'l3_visit':>12}{'l4_conv':>11}")
    for name, row in [("control", c_), ("treatment", t_)]:
        print(f"  {name:<12}{int(row.n):>13,}{int(row.exp_):>13,}{int(row.v):>12,}{int(row.c):>11,}")
    print(f"\n  {'rate':<26}{'control':>12}{'treatment':>12}{'Δ(pp)':>10}{'相对':>10}")
    for label, num in [("exposure rate", "exp_"), ("visit rate", "v"), ("conversion rate", "c")]:
        pc, pt = c_[num] / c_.n, t_[num] / t_.n
        rel = f"{pt/pc-1:>+9.2%}" if pc > 0 else f"{'—':>9}"
        print(f"  {label:<26}{pc:>12.5%}{pt:>12.5%}{(pt-pc)*100:>+10.4f}{rel}")
    print(f"\n  ⚠ 注意：visit|exposure = {t_.v/t_.exp_:.1%} > 100%")
    print(f"     说明访问可在无广告曝光时自然发生，链路不是严格嵌套，"
          f"exposure 是处理依从性而非 visit 的前置条件")
    print(f"  {'conv | visit':<26}{c_.c/c_.v:>12.5%}{t_.c/t_.v:>12.5%}"
          f"{(t_.c/t_.v - c_.c/c_.v)*100:>+10.4f}{ (t_.c/t_.v)/(c_.c/c_.v)-1:>+10.2%}")

    # 正式检验（多重比较：2 个主指标 -> Bonferroni）
    alpha_b = 0.05 / 2
    print(f"\n  ITT 检验（Bonferroni α=0.025，2 个主指标）")
    res = {}
    for label, num in [("visit", "v"), ("conversion", "c")]:
        r = two_prop(int(c_[num]), int(c_.n), int(t_[num]), int(t_.n), alpha_b)
        sig = "显著" if r["p"] < alpha_b else "不显著"
        print(f"    {label:<11} {r['p_c']:.5%} → {r['p_t']:.5%}  "
              f"Δ={r['diff']:+.6f} ({r['rel']:+.2%})  "
              f"95%CI[{r['lo']:+.6f},{r['hi']:+.6f}]  p={fmt_p(r['p'])} {sig}")
        res[label] = {k: float(v) for k, v in r.items()}
    R["itt"] = res
    R["funnel"] = dict(n_c=int(c_.n), n_t=int(t_.n),
                       exposures_c=int(c_.exp_), exposures_t=int(t_.exp_),
                       visits_c=int(c_.v), visits_t=int(t_.v),
                       conv_c=int(c_.c), conv_t=int(t_.c))
    # 实测 MDE
    for label, num in [("visit", "v"), ("conversion", "c")]:
        p1, p2 = c_[num] / c_.n, t_[num] / t_.n
        se = np.sqrt(p1 * (1 - p1) / c_.n + p2 * (1 - p2) / t_.n)
        mde = (stats.norm.ppf(1 - alpha_b / 2) + stats.norm.ppf(0.8)) * se
        print(f"    {label} 可达 MDE = {mde*100:.4f} pp（相对 {mde/p1:.2%}）")
        R.setdefault("mde", {})[label] = dict(abs=float(mde), rel=float(mde / p1))
    return f


# ══════════════════════════════════════════════════════════════
# 4. 依从性诊断 + CACE
# ══════════════════════════════════════════════════════════════
def stage_compliance(con):
    banner("[4] 依从性与曝光内生性（本实验最关键的归因环节）")
    df = con.execute("""
        SELECT
            CASE WHEN treatment=0 THEN 'control'
                 WHEN exposure=1 THEN 'treated_exposed'
                 ELSE 'treated_not_exposed' END AS grp,
            COUNT(*) n, AVG(visit) v, AVG(conversion) c
        FROM raw GROUP BY grp ORDER BY grp
    """).df()
    print(f"  {'组':<22}{'n':>12}{'visit':>10}{'conv':>10}")
    for _, r in df.iterrows():
        print(f"  {r.grp:<22}{int(r.n):>12,}{r.v:>10.4%}{r.c:>10.4%}")

    g = df.set_index("grp")
    ctrl_v, ctrl_c = g.loc["control", "v"], g.loc["control", "c"]
    exp_v, exp_c = g.loc["treated_exposed", "v"], g.loc["treated_exposed", "c"]
    nev_v, nev_c = g.loc["treated_not_exposed", "v"], g.loc["treated_not_exposed", "c"]

    print(f"\n  ▼ 陷阱演示")
    print(f"    exposed vs control：visit {ctrl_v:.2%} → {exp_v:.2%}（{exp_v/ctrl_v:.1f} 倍）")
    print(f"    未曝光组 vs control：visit {ctrl_v:.2%} → {nev_v:.2%}（{nev_v/ctrl_v:.2f} 倍）")
    direction = "更低" if nev_v < ctrl_v else "更高"
    print(f"    → 同样没看到广告的人，被分到 treatment 组后 visit 反而{direction}")
    print(f"      => 曝光分配存在明确的选择效应，exposed vs control 的差值是混杂结果，不是因果效应")

    # ITT / CACE
    n_t = int(g.loc[["treated_exposed", "treated_not_exposed"], "n"].sum())
    n_c = int(g.loc["control", "n"])
    p_exp = g.loc["treated_exposed", "n"] / n_t
    itt_v = exp_v * p_exp + nev_v * (1 - p_exp) - ctrl_v
    itt_c = exp_c * p_exp + nev_c * (1 - p_exp) - ctrl_c
    cace_c = itt_c / p_exp

    print(f"\n  ▼ 三个口径（以 conversion 为例）")
    print(f"    ITT          = {itt_c:+.6f} （相对 {itt_c/ctrl_c:+.1%}）  ← RCT 保证无偏，用于决策")
    # 相对提升一律用 (估计值 − 对照率)/对照率；cace_c 是绝对率，不能直接除以对照率
    print(f"    Wald CACE    = {cace_c:+.6f} （相对 {(cace_c-ctrl_c)/ctrl_c:+.1%}）  ← 需 monotonicity + 排他性")
    print(f"    naive exposed= {exp_c - ctrl_c:+.6f} （相对 {(exp_c-ctrl_c)/ctrl_c:+.1%}）  ← 有偏，不可用")
    print(f"    P(曝光|treatment) = {p_exp:.4%}，即 {1-p_exp:.2%} 的投放预算落在未曝光人群")

    R["compliance"] = dict(
        ctrl=dict(n=n_c, visit=float(ctrl_v), conv=float(ctrl_c)),
        exposed=dict(n=int(g.loc["treated_exposed", "n"]), visit=float(exp_v), conv=float(exp_c)),
        not_exposed=dict(n=int(g.loc["treated_not_exposed", "n"]), visit=float(nev_v), conv=float(nev_c)),
        p_exposure=float(p_exp),
        itt_conv=float(itt_c), cace_conv=float(cace_c), naive_conv=float(exp_c - ctrl_c),
        selection_bias=float(nev_v - ctrl_v),
    )


# ══════════════════════════════════════════════════════════════
# 5. 分层 + 异质性检验
# ══════════════════════════════════════════════════════════════
def stage_strata(con):
    banner("[5] 分层观测实验组差异")
    df = con.execute("""
        SELECT strat_value,
               COUNT(*) FILTER (WHERE treatment=1) n_t,
               COUNT(*) FILTER (WHERE treatment=0) n_c,
               AVG(visit)      FILTER (WHERE treatment=1) v_t,
               AVG(visit)      FILTER (WHERE treatment=0) v_c,
               AVG(conversion) FILTER (WHERE treatment=1) c_t,
               AVG(conversion) FILTER (WHERE treatment=0) c_c
        FROM abt GROUP BY strat_value ORDER BY strat_value
    """).df()
    print(f"  {'分箱':<6}{'n_t':>10}{'n_c':>9}{'visit_c':>9}{'visit_t':>9}{'rel':>9}"
          f"{'conv_c':>9}{'conv_t':>9}{'rel':>9}{'增量转化':>10}")
    total_inc = 0.0
    rows = []
    for _, r in df.iterrows():
        inc = (r.c_t - r.c_c) * r.n_t
        total_inc += inc
        rows.append((r.strat_value, inc))
        print(f"  {int(r.strat_value):<6}{int(r.n_t):>10,}{int(r.n_c):>9,}"
              f"{r.v_c:>9.3%}{r.v_t:>9.3%}{r.v_t/r.v_c-1:>+9.1%}"
              f"{r.c_c:>9.4%}{r.c_t:>9.4%}{r.c_t/r.c_c-1:>+9.1%}{inc:>10.0f}")
    print(f"  {'合计':<6}{'':>10}{'':>9}{'':>9}{'':>9}{'':>9}{'':>9}{'':>9}{total_inc:>10.0f}")

    # 符号一致性
    sign_ok = ((df.c_t - df.c_c) > 0).mean()
    print(f"\n  效应符号一致率 = {sign_ok:.0%}（稳定性口径 #2）")

    # ---------- 层内 SRM：总体通过 ≠ 层内通过 ----------
    # 总体 85:15 的卡方 p=0.999 只说明两臂总量对；分配如果与特征相关
    #（Criteo 公开数据做过非均匀子采样，这是真实风险），层内份额会偏离 85%。
    print(f"\n  ▼ 层内 SRM 检验（逐箱 treatment 份额 vs 设计 85%）")
    chi2_w = 0.0
    zs = []
    for _, r in df.iterrows():
        n = r.n_t + r.n_c
        z = (r.n_t - n * 0.85) / np.sqrt(n * 0.85 * 0.15)
        zs.append(z)
        chi2_w += z * z
        flag = "  ← 偏离" if abs(z) > 5 else ""
        print(f"    箱 {int(r.strat_value)}：{r.n_t/n:.3%}  z={z:+.1f}{flag}")
    p_w = float(stats.chi2.sf(chi2_w, len(df) - 1))
    print(f"    联合卡方 = {chi2_w:,.0f}（df={len(df)-1}），p = {p_w:.1e}")
    if p_w < 1e-6:
        print(f"    → 层内分配与 f0 分箱显著相关：assignment 不是特征无关的。")
        print(f"      后果：高转化箱被过度分入处理组，粗 ITT 会被向上混杂；")
        print(f"      分层加权的调整估计（下方）才是抗混杂的那个。")

    # 分层加权调整 ITT：Σ s_k · (c_tk − c_ck)，s_k 为箱占比（总体加权）
    N = float(df.n_t.sum() + df.n_c.sum())
    s_k = (df.n_t + df.n_c) / N
    itt_adj = float((s_k * (df.c_t - df.c_c)).sum())
    print(f"\n  分层加权调整 ITT = {itt_adj:+.6f}（粗 ITT 见 [4]，两者差即选择混杂的量级）")

    # 正式异质性检验：logit ~ treatment + stratum + treatment×stratum
    # 样本：确定性 fold 0（sql/00_samples.sql，约 280 万行，任何机器重跑一致。
    # 旧版用 reservoir 抽 200 万，同一 seed 两次运行结果不同，p 在 0.019~0.079 漂移）
    d = con.execute("""SELECT treatment, strat_value, conversion FROM sample_lr""").df()
    D = pd.get_dummies(d.treatment, prefix="t", dtype=float)
    S = pd.get_dummies(d.strat_value, prefix="s", drop_first=True, dtype=float)
    X = pd.concat([D[["t_1"]], S], axis=1)
    inter = pd.DataFrame({f"i{i}": D["t_1"].values * S.iloc[:, i].values
                          for i in range(S.shape[1])})
    y = d.conversion.values.astype(float)

    def loglik(cols):
        b = fit_logit(cols.values, y)
        z = np.column_stack([np.ones(len(cols)), cols.values]) @ b
        return np.sum(y * z - np.logaddexp(0, z))

    ll0 = loglik(X)
    ll1 = loglik(pd.concat([X, inter], axis=1))
    lr = 2 * (ll1 - ll0)
    dfree = inter.shape[1]
    pval = stats.chi2.sf(lr, dfree)
    print(f"  异质性 LR 检验：LR={lr:.2f}, df={dfree}, p={fmt_p(pval)} "
          f"-> {'存在显著异质性' if pval < 0.05 else '未见显著异质性'}")

    R["strata"] = dict(n_strata=int(len(df)), sign_consistency=float(sign_ok),
                       lr=float(lr), p=float(pval), total_incremental=float(total_inc),
                       detail=[{k: float(v) for k, v in r.items()} for _, r in df.iterrows()],
                       srm_within=dict(
                           shares=[float(r.n_t / (r.n_t + r.n_c)) for _, r in df.iterrows()],
                           z=[float(v) for v in zs], chi2=float(chi2_w), p=p_w,
                           df=int(len(df) - 1)),
                       itt_adjusted=dict(
                           value=itt_adj,
                           note="Σ s_k·(c_tk−c_ck) 层占比加权；粗 ITT 含 assignment-特征相关的向上混杂，以此为准更保守"))


# ══════════════════════════════════════════════════════════════
# 6. CATE + Qini（抽样建模）
# ══════════════════════════════════════════════════════════════
def stage_cate(con):
    """CATE 与 Qini —— 五折轮转交叉拟合（确定性，样本外）。

    折划分见 sql/00_samples.sql：md5 哈希 % 5，每折约 280 万行。
    每折用 fold i 训练 T-learner，在 fold i+1（模 5）上评估：
      - 14M 用户每一个都会被且只会被评估一次（五对折恰好轮转覆盖全量）
      - 训练 280 万 / 评估 280 万，各是旧版（30 万/30 万）的 9 倍，方差显著更低
      - 五个样本外 Qini 的极差，本身就是「结论稳定性」的直接证据
    另报一个「同折自评」参考值（fold 0 训练 → fold 0 评估），
    它与交叉拟合值的差就是过拟合缺口。

    历史教训：旧版 reservoir 抽 60 万 + 单次 50/50 切分，同一 seed 两次
    运行的样本外 Qini 在 0.05 与 0.50 之间漂移（连样本行数都不稳），
    单次结果不能作为结论；改确定性五折后任何机器重跑数字逐位一致。
    """
    banner("[6] CATE 与 Qini（五折轮转交叉拟合，样本外为准）")

    def load_fold(i):
        d = con.execute(f"SELECT {','.join(FEATS)}, treatment, conversion, exposure "
                        f"FROM det WHERE fold = {i}").df()
        return (d[FEATS].values.astype(float),
                d.conversion.values.astype(float),
                d.treatment.values.astype(int),
                d.exposure.values.astype(float))

    def deciles(y, w, cate):
        """按预测增量降序切 10 段，返回每段 (预测均值, 实际增量, 段大小)"""
        order = np.argsort(-cate)
        yy, ww, cc = y[order], w[order].astype(bool), cate[order]
        n = len(yy)
        segs = []
        for b in np.array_split(np.arange(n), 10):
            yb, wb, cb = yy[b], ww[b], cc[b]
            if wb.sum() == 0 or (~wb).sum() == 0:
                continue
            segs.append((float(cb.mean()),
                         float(yb[wb].mean() - yb[~wb].mean()),
                         int(len(b))))
        return segs

    def decile_exposure(e, cate):
        """与 deciles 同一排序，返回每段的曝光率（两臂合计，对照臂曝光恒为 0）"""
        ee = e[np.argsort(-cate)]
        return [float(ee[b].mean()) for b in np.array_split(np.arange(len(ee)), 10)]

    def qini_of(segs):
        """段级归一化 Qini：2×累计曲线下面积 − 1（段已按预测降序排好）"""
        incs = np.array([s[1] for s in segs])
        sizes = np.array([s[2] for s in segs], dtype=float)
        total = float((incs * sizes).sum())
        if abs(total) < 1e-12:
            return 0.0
        x = np.concatenate([[0.0], np.cumsum(sizes) / sizes.sum()])
        y2 = np.concatenate([[0.0], np.cumsum(incs * sizes) / total])
        return 2 * float(np.trapezoid(y2, x)) - 1

    fold_q = []          # 每折（训练 fold i → 评估 fold i+1）的样本外 Qini
    all_segs = []        # (fold, decile_rank, pred, inc, size) —— 合并曲线用
    all_exp = []         # (fold, decile_rank, exposure_rate, size) —— 曝光率诊断
    ate_pairs = []
    for i in range(5):
        Xtr, ytr, wtr, _ = load_fold(i)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ztr = (Xtr - mu) / sd
        b1 = fit_logit(Ztr[wtr == 1], ytr[wtr == 1])
        b0 = fit_logit(Ztr[wtr == 0], ytr[wtr == 0])

        Xte, yte, wte, ete = load_fold((i + 1) % 5)
        Zte = (Xte - mu) / sd
        Xd = np.column_stack([np.ones(len(Zte)), Zte])
        cate = sigmoid(Xd @ b1) - sigmoid(Xd @ b0)
        segs = deciles(yte, wte, cate)
        q = qini_of(segs)
        fold_q.append(q)
        ate = float(yte[wte == 1].mean() - yte[wte == 0].mean())
        ate_pairs.append(ate)
        exp_rates = decile_exposure(ete, cate)
        for rank, (p_, inc, sz) in enumerate(segs):
            all_segs.append((i, rank, p_, inc, sz))
            all_exp.append((i, rank, exp_rates[rank], sz))
        print(f"  fold {i} 训练 → fold {(i+1)%5} 评估：Qini = {q:+.4f}   "
              f"评估折 ATE = {ate:+.6f}  n = {len(yte):,}")
        del Xtr, ytr, wtr, Ztr, Xte, yte, wte, ete, Zte, Xd, cate

    # 同折自评（过拟合参考）：fold 0 模型 → fold 0 自评
    Xtr, ytr, wtr, _ = load_fold(0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr = (Xtr - mu) / sd
    b1 = fit_logit(Ztr[wtr == 1], ytr[wtr == 1])
    b0 = fit_logit(Ztr[wtr == 0], ytr[wtr == 0])
    Xd = np.column_stack([np.ones(len(Ztr)), Ztr])
    cate_in = sigmoid(Xd @ b1) - sigmoid(Xd @ b0)
    q_in = qini_of(deciles(ytr, wtr, cate_in))
    del Xtr, ytr, wtr, Ztr, Xd, cate_in

    # ---------- 合并口径：50 段（5 折 × 10 分位）按预测值重排 ----------
    all_segs.sort(key=lambda s: -s[2])
    pooled_q = qini_of([(s[2], s[3], s[4]) for s in all_segs])
    total_inc = sum(s[3] * s[4] for s in all_segs)
    # 合并分位表：把每折各自的第 k 分位聚在一起（各 5 段）
    by_rank = {}
    for i, rank, p_, inc, sz in all_segs:
        by_rank.setdefault(rank, []).append((p_, inc, sz))
    table = []
    for rank in range(10):
        rows = by_rank[rank]
        wsum = sum(sz for _, _, sz in rows)
        pred = sum(p * sz for p, _, sz in rows) / wsum
        inc = sum(v * sz for _, v, sz in rows) / wsum
        table.append((pred, inc, wsum))

    print(f"\n  各折 Qini：min {min(fold_q):+.4f} / mean {np.mean(fold_q):+.4f} / max {max(fold_q):+.4f}"
          f"（极差 {max(fold_q)-min(fold_q):+.4f} = 稳定性证据）")
    print(f"  合并样本外 Qini（50 段重排）= {pooled_q:+.4f}；同折自评参考值 = {q_in:+.4f}")
    print(f"  （自评 − 交叉拟合 = {q_in-pooled_q:+.4f}，这部分是过拟合）")

    print(f"\n  {'分位':<8}{'预测增量':>13}{'实际增量':>13}{'实际/预测':>12}{'累计占比':>12}")
    cum = 0.0
    for rank, (pred, inc, sz) in enumerate(table):
        cum += inc * sz
        ratio = inc / pred if abs(pred) > 1e-9 else float("nan")
        print(f"  前{(rank+1)*10:>3}%  {pred:>13.6f}{inc:>13.6f}{ratio:>12.2f}"
              f"{cum/total_inc:>12.1%}")

    top10 = float(sum(p_ * sz for p_, inc, sz in by_rank[0]) /
                  sum(sz for _, _, sz in by_rank[0]))
    top10_cum = float(sum(inc * sz for _, inc, sz in by_rank[0]) / (total_inc))
    top20_cum = float((sum(inc * sz for _, inc, sz in by_rank[0]) +
                       sum(inc * sz for _, inc, sz in by_rank[1])) / total_inc)
    print(f"\n  样本外 top10 实际增量 = {top10:+.6f}；累计贡献 前10% = {top10_cum:.1%}，前20% = {top20_cum:.1%}")

    # ---------- 曝光率诊断：模型是在挑「会被打动的人」还是「会被曝光的人」 ----------
    exp_by_rank = {}
    for i, rank, rate, sz in all_exp:
        exp_by_rank.setdefault(rank, []).append((rate, sz))
    dec_exp = []
    for rank in range(10):
        rows = exp_by_rank[rank]
        wsum = sum(sz for _, sz in rows)
        dec_exp.append(float(sum(r * sz for r, sz in rows) / wsum))
    overall_exp = float(sum(r * sz for _, _, r, sz in all_exp) / sum(sz for _, _, _, sz in all_exp))
    # 按曝光率折算的每曝光用户增量（Wald CACE 的分位版本）：inc_k / P(exposure|trt, decile k)
    # 对照臂曝光恒为 0，故 P(exposure|trt) = 两臂合计曝光率 / treated 占比
    share_t = float(con.execute("SELECT AVG(treatment) FROM det").fetchone()[0])
    dec_cace = [float(t[1] * share_t / e) if e > 0 else float("nan")
                for t, e in zip(table, dec_exp)]
    print(f"\n  曝光率诊断（两臂合计；对照组曝光恒为 0，故数值 ≈ {share_t:.2%} × 处理组曝光率）")
    print(f"  全体曝光率 = {overall_exp:.4%}")
    print(f"  {'分位':<12}{'曝光率':>10}{'相对全体':>10}{'折算每曝光用户增量':>20}")
    for rank, r in enumerate(dec_exp):
        lift = r / overall_exp if overall_exp > 0 else float("nan")
        print(f"  第{rank*10+1:>3}–{(rank+1)*10:>3}%  {r:>10.4%}{lift:>9.2f}x{dec_cace[rank]:>20.4%}")

    R["cate"] = dict(
        protocol="5 折轮转交叉拟合（md5 确定性分折，见 sql/00_samples.sql）",
        fold_qini=[float(v) for v in fold_q],
        qini_out=float(pooled_q), qini_insample_ref=float(q_in),
        ate_out=float(np.mean(ate_pairs)),
        # 分位等权 g-computation：Σ_k (ȳ_tk−ȳ_ck)/10。各分位处理份额不等时，
        # 它与 ate_out（全体直接算差）不相等——差值就是 assignment-特征相关的量级。
        ate_decile=float(np.mean([t[1] for t in table])),
        top10_cum_out=top10_cum, top20_cum_out=top20_cum, top10_out=top10,
        decile_pred=[t[0] for t in table], decile_inc=[t[1] for t in table],
        decile_exposure=dec_exp, overall_exposure=overall_exp, decile_cace=dec_cace,
        treated_share=share_t,
        exposure_note="两臂合计曝光率，按 CATE 样本外分位；对照臂 exposure 恒为 0，"
                      "故该值 ≈ treated 占比 × 处理组内曝光率；decile_cace 沿用 Wald CACE 的排他性假设",
        qini_note="归一化 Qini = 2×曲线面积−1，不可与文献原始 Qini 直接比较")

    # 供 figures.py / cate_check.py 复用：曲线与分位表落盘（图不再自己拟合）
    import json as _json
    with open("report/cate_folds.json", "w", encoding="utf-8") as fh:
        _json.dump(dict(
            protocol=R["cate"]["protocol"], fold_qini=R["cate"]["fold_qini"],
            qini_out=R["cate"]["qini_out"], qini_insample_ref=R["cate"]["qini_insample_ref"],
            ate_decile=R["cate"]["ate_decile"],
            top10_cum_out=top10_cum, top20_cum_out=top20_cum,
            decile_pred=R["cate"]["decile_pred"], decile_inc=R["cate"]["decile_inc"],
            decile_exposure=R["cate"]["decile_exposure"], overall_exposure=R["cate"]["overall_exposure"],
            decile_cace=R["cate"]["decile_cace"], treated_share=R["cate"]["treated_share"],
            curve_x=[round(v, 4) for v in np.linspace(0, 1, 51)],
            curve_y=[round(v, 4) for v in np.concatenate(
                [[0.0], np.cumsum([s[3] * s[4] for s in all_segs]) / total_inc])],
        ), fh, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════
# 7. 收益验证
# ══════════════════════════════════════════════════════════════
def stage_roi(aov, margin):
    """收益验证。

    口径说明（重要）：
    Criteo 数据每行是一个 user，exposure 是「该用户是否至少被曝光一次」的
    用户级 0/1 变量，**不是曝光次数**。所以：
      - 无法算总曝光次数，也就无法用 CPM × 次数 算绝对成本
      - 无法算绝对 ROI
    能做的是反向推导：要让这笔投放不亏，每个「被曝光用户」的成本上限是多少。
    这个上限是决策有用的，而且不依赖我们拿不到的数据。
    """
    banner("[7] 收益验证：每曝光用户的成本上限（反向推导）")
    f = R["funnel"]
    n_t, n_c = f["n_t"], f["n_c"]
    inc_conv = n_t * (f["conv_t"] / n_t - f["conv_c"] / n_c)
    exposed_users = f["exposures_t"]          # 被曝光的用户数，不是曝光次数
    gross_per_conv = aov * margin
    inc_gross = inc_conv * gross_per_conv

    print("  ⚠ 数据限制：每行是一个 user，exposure 是用户级 0/1 标志")
    print("     → 没有曝光次数、没有单次曝光成本，绝对 CPM / ROI 无法从数据推导")
    print("     → 下面只做反向推导：盈亏平衡时，每个被曝光用户最多能花多少")
    print()
    print(f"  参数假设：AOV={aov} 元  毛利率={margin:.0%}  单均毛利={gross_per_conv:.2f} 元")
    print(f"  增量转化     = {inc_conv:,.0f} 单")
    print(f"  增量毛利     = {inc_gross:,.0f} 元")
    print(f"  被曝光用户   = {exposed_users:,} 人（user 级，不是曝光次数）")
    cap_per_user = inc_gross / exposed_users
    print(f"  >>> 盈亏平衡的「每曝光用户成本」上限 = {cap_per_user:,.2f} 元/人")
    print(f"      超过这个数就亏。这是给运营的投放单价天花板。")
    print()

    print(f"  反过来：如果每曝光用户成本是 C 元，则总成本 = C × {exposed_users:,}")
    print(f"  {'每用户成本':>12}{'总成本':>16}{'CPA':>12}{'ROI':>10}{'结论':>8}")
    for c in [0.5, 1.0, 2.0, 4.0, 8.0]:
        cc = exposed_users * c
        cp = cc / inc_conv if inc_conv > 0 else np.inf
        rr = inc_gross / cc if cc > 0 else np.nan
        print(f"  {c:>12.1f}{cc:>16,.0f}{cp:>12,.2f}{rr:>10.2f}"
              f"{'可行' if rr > 1 else '亏损':>8}")

    print(f"\n  口径提醒：增量转化的分母是全部 {n_t:,} 个 treated 用户，")
    print(f"            不是 {exposed_users:,} 个被曝光用户——因为曝光本身受依从性影响。")

    R["roi"] = dict(aov=float(aov), margin=float(margin), inc_conv=float(inc_conv),
                    inc_gross=float(inc_gross), exposed_users=int(exposed_users),
                    cap_per_user=float(cap_per_user), gross_per_conv=float(gross_per_conv))


# ══════════════════════════════════════════════════════════════
# 8. 稳定性
# ══════════════════════════════════════════════════════════════
def stage_stability(con):
    banner("[8] 稳定性评估（5 个口径）")
    # #1 Bootstrap（在聚合粒度上重采样，避免全量复制）
    # 注意：这里是在一个 150 万行子样本上做的，点估计和主 ITT 会有差异，
    # 报告里必须明确「这是子样本稳定性，不是主估计的置信区间」。
    d = con.execute("""
        SELECT treatment, conversion, visit FROM sample_boot
    """).df()
    y = d.conversion.values.astype(float)
    w = d.treatment.values.astype(bool)
    point = y[w].mean() - y[~w].mean()
    rng = np.random.default_rng(7)
    n = len(y)
    est = np.empty(400)
    for b in range(400):
        idx = rng.integers(0, n, n)
        yb, wb = y[idx], w[idx]
        est[b] = yb[wb].mean() - yb[~wb].mean()
    lo, hi = np.percentile(est, [2.5, 97.5])
    print(f"  #1 抽样稳定性 Bootstrap(B=400，1.5M 子样本)：点估计 {point:+.6f}  "
          f"CI[{lo:+.6f},{hi:+.6f}]  标准误={est.std():.6f}  CV={est.std()/abs(point):.1%}")
    print("      ⚠ 这是 1.5M 子样本的稳定性；点估计与主 ITT 的差异、以及约 3 倍于解析 SE 的宽度，")
    print("        来自子样本，不是全量 13.98M 的置信区间。")

    # #2 人群稳定性（来自 strata）
    s = R["strata"]
    print(f"  #2 人群稳定性：{s['n_strata']} 个分箱效应符号一致率 {s['sign_consistency']:.0%}  "
          f"LR p={fmt_p(s['p'])}")

    # #3 泛化稳定性：fold 2 内按 md5key 末位奇偶对半（确定性），比较两半效应
    d2 = con.execute("""
        SELECT treatment, conversion,
               (CAST('0x' || substr(md5key, 32, 1) AS UBIGINT) % 2) AS half
        FROM det WHERE fold = 2
    """).df()
    diffs = []
    for h in [0, 1]:
        sub = d2[d2.half == h]
        a = sub[sub.treatment == 1].conversion.mean()
        b_ = sub[sub.treatment == 0].conversion.mean()
        diffs.append(a - b_)
    gap = abs(diffs[0] - diffs[1])
    print(f"  #3 泛化稳定性：两半效应 {diffs[0]:+.6f} vs {diffs[1]:+.6f}，"
          f"绝对差 {gap:.6f}（相对差 {gap/abs(np.mean(diffs)):.1%}）"
          f" -> {'稳定' if gap/abs(np.mean(diffs)) < 0.3 else '波动偏大'}")

    # #4 决策稳定性：给定投放单价，需要多高毛利率才不亏
    roi = R["roi"]
    # 新口径：盈亏平衡 → 每曝光用户成本 × 曝光用户数 = 增量转化 × AOV × 毛利率
    # 所以给定每用户成本 C，所需毛利率 = C × exposed_users / (inc_conv × AOV)
    for c in [1.0, 2.0]:
        need = c * roi["exposed_users"] / (roi["inc_conv"] * roi["aov"])
        verdict = "当前毛利率够" if roi["margin"] >= need else "当前毛利率不够"
        print(f"  #4 决策稳定性：每曝光用户成本 {c:.1f} 元时，毛利率需 > {need:.2%}"
              f"（当前 {roi['margin']:.0%}，{verdict}）")

    # #5 提前看数风险
    n_max, n_sim, n_checks = 20000, 500, 30
    rg = np.random.default_rng(9)
    p0 = 0.0019
    a = (rg.random((n_sim, n_max)) < p0).cumsum(1)
    b_ = (rg.random((n_sim, n_max)) < p0).cumsum(1)
    ks = np.linspace(500, n_max, n_checks).astype(int)
    x1, x2 = a[:, ks - 1].astype(float), b_[:, ks - 1].astype(float)
    nn = ks[None, :].astype(float)
    pp = (x1 + x2) / (2 * nn)
    se = np.sqrt(pp * (1 - pp) * 2 / nn)
    z = np.abs(x2 / nn - x1 / nn) / np.maximum(se, 1e-12)
    rate = float((z > 1.96).any(axis=1).mean())
    print(f"  #5 提前看数风险：模拟 {n_sim} 次（真效应=0），至少一次假显著 = {rate:.1%}"
          f"（理论 5%，膨胀 {rate/0.05:.1f}x）")

    R["stability"] = dict(boot_point=float(point), boot_lo=float(lo), boot_hi=float(hi),
                          boot_se=float(est.std()), boot_n=int(n),
                          sign_consistency=s["sign_consistency"],
                          half_gap=float(gap), peeking_rate=float(rate))


# ══════════════════════════════════════════════════════════════
class Tee:
    """同时写 stdout 和文件。

    为什么需要：以前 run_log.txt 靠 shell 的 `>` 重定向生成，
    跑脚本忘了重定向时，报告就停留在上一次运行的结果上——
    这是 README 数字和 run_log 长期对不上的直接原因。
    现在脚本自己写日志，跑一次同步一次。
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aov", type=float, default=200.0)
    ap.add_argument("--margin", type=float, default=0.30)
    ap.add_argument("--skip-cate", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    print(f"数据：{DATA}")
    con = duck_connect()
    for f in ["sql/01_build.sql", "sql/00_samples.sql"]:
        con.execute(open(f, encoding="utf-8").read())
        print(f"已执行 {f}")

    # 脚本自己写 run_log，避免遗忘重定向导致日志与 README 脱节
    _stdout = sys.stdout
    logf = open("report/run_log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(_stdout, logf)
    try:
        stage_srm(con)
        stage_aa(con)
        stage_itt(con)
        stage_compliance(con)
        stage_strata(con)
        if not a.skip_cate:
            stage_cate(con)
        stage_roi(a.aov, a.margin)
        stage_stability(con)
    finally:
        sys.stdout = _stdout
        logf.close()

    os.makedirs("report", exist_ok=True)
    with open("report/metrics.json", "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=2)
    print(f"\n{'='*74}\n全部完成，耗时 {time.time()-t0:.1f}s，结果已存 report/metrics.json")


if __name__ == "__main__":
    main()
