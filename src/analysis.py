"""
Criteo-UPLIFT v2.1 —— 广告增量性 A/B 实验 · 分析层
====================================================
对应 JD：实验方案设计 → 分层观测 → 链路指标 → SQL/Python 归因 → 收益验证 → 稳定性评估

运行：
    python src/analysis.py                 # 全量
    python src/analysis.py --sample 2000000  # 抽样（调试用）
    python src/analysis.py --aov 200 --margin 0.3

输出：终端报告 + report/metrics.json
"""
import argparse
import json
import os
import sys
import time

import duckdb
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
    """L2 正则的 logistic 回归（L-BFGS），避免依赖 statsmodels"""
    Xd = np.column_stack([np.ones(len(X)), X])

    def nll(b):
        z = Xd @ b
        return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(b[1:] ** 2)

    res = optimize.minimize(nll, np.zeros(Xd.shape[1]), method="L-BFGS-B",
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
# 2. AA 检验
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
        mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se
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
    print(f"    Wald CACE    = {cace_c:+.6f} （相对 {cace_c/ctrl_c:+.1%}）  ← 需 monotonicity + 排他性")
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

    # 正式异质性检验：logit ~ treatment + stratum + treatment×stratum
    d = con.execute("""SELECT treatment, strat_value, conversion FROM abt
                       WHERE hash(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,treatment,conversion,visit,exposure) % 10000 < 1431""").df()
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
                       detail=[{k: float(v) for k, v in r.items()} for _, r in df.iterrows()])


# ══════════════════════════════════════════════════════════════
# 6. CATE + Qini（抽样建模）
# ══════════════════════════════════════════════════════════════
def stage_cate(con, n_sample=600_000, seed=42):
    """CATE 与 Qini。

    口径说明：
      - 数据切 50/50：一半拟合、一半评估。样本内指标会系统性高估，
        所以下面同时报「样本内」和「样本外」两组，以样本外为准。
      - Qini 用的是 2×曲线面积−1 的**归一化** Qini，取值范围 [−1, 1]，
        和文献里常见的原始 Qini 系数（量纲取决于转化率）不可直接比较。
      - 随机重排检验只能说明「比随机排序好」，不能说明「没记住样本」。
        泛化能力要看样本外那一组数。
    """
    banner("[6] CATE 与 Qini（train/test 分割，样本外为准）")
    d = con.execute(f"""
        SELECT * FROM (
            SELECT {','.join(FEATS)}, treatment, conversion, visit
            FROM raw WHERE hash(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,treatment,conversion,visit,exposure) % 10000 < 430
        )
    """).df()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(d))
    d = d.iloc[idx].reset_index(drop=True)
    n_tr = len(d) // 2
    tr, te = d.iloc[:n_tr], d.iloc[n_tr:]

    # 拟合用训练集；样本内报告在训练集上算一次，样本外在测试集上算一次
    # 注意：评估要复用训练集的标准化参数和系数，所以这里手写第二遍
    Xtr = tr[FEATS].values.astype(float)
    ytr = tr.conversion.values.astype(float)
    wtr = tr.treatment.values.astype(int)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    b1 = fit_logit(((Xtr - mu) / sd)[wtr == 1], ytr[wtr == 1])
    b0 = fit_logit(((Xtr - mu) / sd)[wtr == 0], ytr[wtr == 0])

    def evaluate(sub, tag):
        X = sub[FEATS].values.astype(float)
        y = sub.conversion.values.astype(float)
        w = sub.treatment.values.astype(int)
        Xs = (X - mu) / sd                       # 用训练集的 mu/sd
        Xd = np.column_stack([np.ones(len(Xs)), Xs])
        cate = sigmoid(Xd @ b1) - sigmoid(Xd @ b0)
        ate = y[w == 1].mean() - y[w == 0].mean()
        order = np.argsort(-cate)
        yy, ww = y[order], w[order].astype(bool)
        n = len(yy)
        incs, sizes = [], []
        for b in np.array_split(np.arange(n), 10):
            yb, wb = yy[b], ww[b]
            if wb.sum() == 0 or (~wb).sum() == 0:
                continue
            incs.append(yb[wb].mean() - yb[~wb].mean())
            sizes.append(len(b))
        incs, sizes = np.array(incs), np.array(sizes)
        cum = np.cumsum(incs * sizes)
        total = cum[-1] if abs(cum[-1]) > 1e-12 else 1e-12
        x = np.concatenate([[0.0], np.cumsum(sizes) / n])
        y2 = np.concatenate([[0.0], cum / total])
        qini_norm = 2 * float(np.trapezoid(y2, x)) - 1
        return dict(tag=tag, n=n, ate=float(ate), qini_norm=qini_norm, incs=incs,
                    sizes=sizes, cate=cate, order=order, total=total)

    in_s = evaluate(tr, "样本内（训练集）")
    out_s = evaluate(te, "样本外（测试集）")

    print(f"  总样本 {len(d):,} → 训练 {len(tr):,} / 测试 {len(te):,}")
    print(f"\n  {'':<18}{'Qini(归一化)':>14}{'ATE':>13}{'前10%累计':>12}{'前20%累计':>12}")
    for m in (in_s, out_s):
        cum = np.cumsum(m["incs"] * m["sizes"])
        c10 = float(cum[0] / m["total"]) if len(m["incs"]) else 0
        c20 = float(cum[1] / m["total"]) if len(m["incs"]) >= 2 else 0
        print(f"  {m['tag']:<18}{m['qini_norm']:>14.4f}{m['ate']:>13.6f}{c10:>12.1%}{c20:>12.1%}")

    m = out_s
    print(f"\n  以样本外为准：Qini(归一化) = {m['qini_norm']:+.4f}")
    print(f"  （样本内 {in_s['qini_norm']:+.4f} 比样本外高 {in_s['qini_norm']-m['qini_norm']:+.4f}，"
          f"这部分就是过拟合）")
    print(f"\n  {'分位':<8}{'样本':>10}{'预测增量':>13}{'实际增量':>13}{'累计占比':>12}")
    cumshare = 0.0
    for i, (b, inc, sz) in enumerate(zip(np.array_split(np.arange(m["n"]), len(m["incs"])),
                                          m["incs"], m["sizes"])):
        cumshare += inc * sz / m["total"]
        pred = m["cate"][m["order"]][b].mean()
        print(f"  前{(i+1)*10:>3}%  {sz:>10,}{pred:>13.6f}{inc:>13.6f}{cumshare:>12.1%}")

    top20 = float(m["incs"][:2].mean())
    bot20 = float(m["incs"][-2:].mean())
    print(f"\n  样本外最优 20% 增量 = {top20:+.6f} | 最差 20% = {bot20:+.6f}")
    if bot20 <= 0:
        print(f"  ⚠ 最差分组出现负增量——说明模型把一部分被广告伤害的人排到了末尾，")
        print(f"    这是好事（说明排序有信息），但也意味着「Top20 贡献 90%+」这类说法要谨慎。")

    R["cate"] = dict(n_total=int(len(d)), n_train=int(len(tr)), n_test=int(len(te)),
                     qini_in=float(in_s["qini_norm"]), qini_out=float(m["qini_norm"]),
                     ate_in=float(in_s["ate"]), ate_out=float(m["ate"]),
                     top20_out=top20, bot20_out=bot20,
                     qini_note="归一化 Qini = 2×曲线面积−1，不可与文献原始 Qini 直接比较")


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
        SELECT treatment, conversion, visit FROM raw WHERE hash(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,treatment,conversion,visit,exposure) % 10000 < 1073
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
    print(f"  #1 抽样稳定性 Bootstrap(B=400)：点估计 {point:+.6f}  "
          f"CI[{lo:+.6f},{hi:+.6f}]  标准误={est.std():.6f}  CV={est.std()/abs(point):.1%}")

    # #2 人群稳定性（来自 strata）
    s = R["strata"]
    print(f"  #2 人群稳定性：{s['n_strata']} 个分箱效应符号一致率 {s['sign_consistency']:.0%}  "
          f"LR p={fmt_p(s['p'])}")

    # #3 泛化稳定性：随机对半切，比较两半效应
    d2 = con.execute("""
        SELECT treatment, conversion, (row_number() OVER () % 2) AS half
        FROM raw WHERE hash(f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,treatment,conversion,visit,exposure) % 10000 < 1431
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
                          boot_se=float(est.std()), sign_consistency=s["sign_consistency"],
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
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    for f in ["sql/01_build.sql"]:
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
